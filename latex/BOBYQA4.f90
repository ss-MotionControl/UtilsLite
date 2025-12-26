!*****************************************************************************************
!>
!  BOBYQA: **B**ound **O**ptimization **BY** **Q**uadratic **A**pproximation
!
!  The purpose of BOBYQA is to seek the least value of a function F of several
!  variables, when derivatives are not available. The constraints are the lower
!  and upper bounds on every variable, which can be set to huge values for
!  unconstrained variables.
!
!  The algorithm is intended to change the variables to values that are close
!  to a local minimum of F. The user, however, should assume responsibility for
!  finding out if the calculations are satisfactory, by considering carefully
!  the values of F that occur.
!
!# References
!  * "[The BOBYQA algorithm for bound constrained optimization without
!    derivatives](http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2009_06.pdf)".
!
!# History
!  * M.J.D. Powell (January 5th, 2009) -- There are no restrictions on or charges
!    for the use of the software. I hope that the time and effort I have spent on
!     developing the package will be helpful to much research and to many applications.
!  * Jacob Williams, July 2015 : refactoring of the code into modern Fortran.

module kind_module
   implicit none
   private
   integer, parameter, public :: wp = kind(1.0d0)  !! Double precision kind
end module kind_module

module bobyqa_module

   use kind_module, only: wp

   real(wp), dimension(:, :), allocatable         :: m_xpt   ! interpolation points
   real(wp), dimension(:, :), allocatable, target :: m_B     ! last n columns of H
   real(wp), dimension(:, :), allocatable, target :: m_Z     ! factorization matrix
   real(wp), dimension(:, :), allocatable :: m_HQ    ! CAMBIATO: matrice piena n x n (solo upper triangular)
   real(wp), dimension(:), allocatable :: m_x_base
   real(wp), dimension(:), allocatable :: m_x_new
   real(wp), dimension(:), allocatable :: m_g_new
   real(wp), dimension(:), allocatable :: m_x_opt
   real(wp), dimension(:), allocatable :: m_g_opt
   real(wp), dimension(:), allocatable :: m_f_val
   real(wp), dimension(:), allocatable, target :: m_v_lag
   real(wp), dimension(:), allocatable :: m_g_lag
   real(wp), dimension(:), allocatable :: m_pq
   real(wp), dimension(:), allocatable :: m_x_alt

   real(wp), dimension(:), allocatable :: m_s_lower
   real(wp), dimension(:), allocatable :: m_s_upper
   real(wp), dimension(:), allocatable :: m_x_lower
   real(wp), dimension(:), allocatable :: m_x_upper

   real(wp), dimension(:), allocatable :: m_xbdi

   real(wp), dimension(:), allocatable :: m_s
   real(wp), dimension(:), allocatable :: m_hs
   real(wp), dimension(:), allocatable :: m_hred

   real(wp) :: m_rho
   real(wp) :: m_rhobeg
   real(wp) :: m_rhoend
   real(wp) :: m_alpha
   real(wp) :: m_beta
   real(wp) :: m_cauchy
   real(wp) :: m_denom
   real(wp) :: m_adelt
   real(wp) :: m_delta
   real(wp) :: m_dsq
   real(wp) :: m_crvmin
   real(wp) :: m_x_opt_square
   real(wp) :: m_ratio
   real(wp) :: m_dnorm
   real(wp) :: m_distsq
   real(wp) :: m_fsave

   real(wp) :: m_diffa, m_diffb, m_diffc

   real(wp), parameter :: m_zero = 0.0_wp
   real(wp), parameter :: m_half = 0.5_wp
   real(wp), parameter :: m_one = 1.0_wp
   real(wp), parameter :: m_two = 2.0_wp
   real(wp), parameter :: m_ten = 10.0_wp
   real(wp), parameter :: m_tenth = 0.1_wp
   real(wp), parameter :: m_eps = 1e-20

   integer :: m_ndim
   integer :: m_maxfun
   integer :: m_knew
   integer :: m_kopt
   integer :: m_iprint
   integer :: m_nv
   integer :: m_npt
   integer :: m_nptm
   integer :: m_nf
   integer :: m_kbase
   integer :: m_ntrits
   integer :: m_nresc
   integer :: m_itest
   integer :: m_nf_saved

   private

   abstract interface
      subroutine func(n, x, f)  !! calfun interface
         import :: wp
         implicit none
         integer, intent(in) :: n
         real(wp), dimension(:), intent(in) :: x
         real(wp), intent(out) :: f
      end subroutine func
   end interface

   public :: bobyqa
   public :: test_rosenbrock
   public :: test_freudenstein_roth
   public :: test_beale
   public :: test_flat_valley
   public :: test_kinks
   public :: test_arglin

contains

   subroutine info(name)

      integer :: i, j
      character*(*) :: name

      return

      write (*, *)
      write (*, *) name

      write (*, '(A,G15.8)') 'm_nf           = ', m_nf
      write (*, '(A,I6)') 'm_knew         = ', m_knew
      write (*, '(A,G15.8)') 'm_rho          = ', m_rho
      write (*, '(A,G15.8)') 'm_crvmin       = ', m_crvmin
      write (*, '(A,G15.8)') 'm_dsq          = ', m_dsq
      write (*, '(A,G15.8)') 'm_alpha        = ', m_alpha
      write (*, '(A,G15.8)') 'm_beta         = ', m_beta
      write (*, '(A,G15.8)') 'm_delta        = ', m_delta
      write (*, '(A,G15.8)') 'm_adelt        = ', m_adelt
      write (*, '(A,G15.8)') 'm_cauchy       = ', m_cauchy
      write (*, '(A,G15.8)') 'm_denom        = ', m_denom
      write (*, '(A,G15.8)') 'm_x_opt_square = ', m_x_opt_square
      write (*, '(A,G15.8)') 'm_distsq       = ', m_distsq

      write (*, *) 'B ='
      do i = 1, size(m_B, 1)
         write (*, '(*(G15.8))') (m_B(i, j), j=1, size(m_B, 2))
      end do

      write (*, *) 'Z ='
      do i = 1, size(m_Z, 1)
         write (*, '(*(G15.8))') (m_Z(i, j), j=1, size(m_Z, 2))
      end do
      write (*, *) 'm_x_new ='
      write (*, '(*(G15.8))') (m_x_new(i), i=1, m_nv)
      write (*, *) 'm_g_new ='
      write (*, '(*(G15.8))') (m_g_new(i), i=1, m_nv)

   end subroutine info

   function dHd(d)
      real(wp) :: d(m_nv), dHd
      ! Contributo lineare
      dHd = dot_product(d, m_g_opt + 0.5*matmul(m_HQ, d))
      if (m_nf > m_npt) then
         dHd = dHd + 0.5*dot_product(d, matmul(m_xpt, m_pq*matmul(transpose(m_xpt), d)))
      end if
   end function dHd

   function grad_times_vector(v) result(Hv)
      real(wp), intent(in) :: v(:)
      real(wp) :: Hv(m_nv)
      Hv = matmul(m_HQ, v)
      if (m_nf > m_npt) then
         Hv = Hv + matmul(m_xpt, m_pq*matmul(transpose(m_xpt), v))
      end if
   end function grad_times_vector

   subroutine swap(a, b)
      real(wp), intent(inout) :: a, b
      real(wp) :: tmp
      tmp = a; a = b; b = tmp
   end subroutine swap

   function squaredNorm(x) result(nrm)
      real(wp), intent(in) :: x(:)
      real(wp) :: nrm
      nrm = sum(x**2)
   end function squaredNorm

   subroutine box_step(s, d, iact, minstep)
      real(wp), intent(in) :: s(1:m_nv), d(1:m_nv)
      real(wp), intent(out) :: minstep
      integer, intent(out) :: iact
      real(wp) :: temp_vec(1:m_nv)

      where (s > 0.0_wp)
         temp_vec = (m_s_upper - m_x_opt - d)/s
      elsewhere(s < 0.0_wp)
         temp_vec = (m_s_lower - m_x_opt - d)/s
      elsewhere
         temp_vec = huge(0.0_wp)
      end where

      iact = minloc(temp_vec, dim=1)
      minstep = temp_vec(iact)
   end subroutine box_step

   subroutine box_adjust(x)
      real(wp) :: x(1:m_nv)
      ! 1. Calcolo distanze iniziali
      m_s_lower = m_x_lower - x
      m_s_upper = m_x_upper - x

      where (m_s_lower >= -m_rhobeg)
         ! --- Caso: Troppo vicino al limite INFERIORE ---
         ! Se fuori, schiaccia sul bordo; se troppo vicino, sposta avanti di rhobeg
         x = merge(m_x_lower, m_x_lower + m_rhobeg, m_s_lower >= 0.0_wp)
         m_s_lower = merge(0.0_wp, -m_rhobeg, m_s_lower >= 0.0_wp)
         ! Ricalcola la distanza dal limite opposto in modo diretto
         m_s_upper = max(m_x_upper - x, m_rhobeg)

      elsewhere(m_s_upper <= m_rhobeg)
         ! --- Caso: Troppo vicino al limite SUPERIORE --- (Speculare al precedente)
         ! Se fuori, schiaccia sul bordo; se troppo vicino, sposta indietro di rhobeg
         x = merge(m_x_upper, m_x_upper - m_rhobeg, m_s_upper <= 0.0_wp)
         m_s_upper = merge(0.0_wp, m_rhobeg, m_s_upper <= 0.0_wp)
         ! Ricalcola la distanza dal limite opposto in modo diretto
         m_s_lower = min(m_x_lower - x, -m_rhobeg)

      end where
   end subroutine box_adjust

   subroutine box_project(x)
      real(wp) :: x(1:m_nv)
      ! 1. Calcolo distanze iniziali
      x = max( min( x, m_x_upper ), m_x_lower )
   end subroutine box_project

   subroutine box_project_s(x)
      real(wp) :: x(1:m_nv)
      ! 1. Calcolo distanze iniziali
      x = max( min( x, m_s_upper ), m_s_lower )
   end subroutine box_project_s

   subroutine model_shift_base()

      implicit none

      real(wp) :: sumpq, fracsq
      real(wp) :: W0(m_nv), W1(m_nv), W2(m_npt), W3(m_npt)
      integer  :: i, j, k

      ! --- Dichiarazione Alias (Puntatori) ---
      real(wp), pointer :: B0(:, :), B1(:, :)

      ! --- Associazione Alias ---
      B0 => m_B(1:m_nv, 1:m_npt)
      B1 => m_B(1:m_nv, m_npt + 1:m_npt + m_nv)

      fracsq = 0.25_wp*m_x_opt_square
      sumpq = sum(m_pq)
      W3 = matmul(transpose(m_xpt), m_x_opt) - m_half*m_x_opt_square
      do k = 1, m_npt
         block
            real(wp) :: tempa, tempb
            tempa = W3(k)
            tempb = fracsq - m_half*tempa
            W0 = m_B(:, k)
            W1 = tempa*m_xpt(:, k) + tempb*m_x_opt
            do i = 1, m_nv
               B1(1:i, i) = B1(1:i, i) + W0(i)*W1(1:i) + W1(i)*W0(1:i)
            end do
         end block
      end do
      !
      !     Then the revisions of BMAT that depend on ZMAT are calculated.
      !
      block
         real(wp) :: temp, sumz, sumw
         do j = 1, m_nptm
            W2 = W3*m_Z(:, j)
            sumz = sum(m_Z(:, j))
            sumw = sum(W2)
            temp = fracsq*sumz - m_half*sumw
            do i = 1, m_nv
               W0(i) = temp*m_x_opt(i) + sum(W2*m_xpt(i, :))
               B0(i, :) = B0(i, :) + W0(i)*m_Z(:, j)
               B1(1:i, i) = B1(1:i, i) + W0(i)*W0(1:i)
            end do
         end do
      end block
      !
      !     The following instructions complete the shift, including the changes
      !     to the second derivative parameters of the quadratic model.
      !
      ! W = Xpt * pq - 0.5 * sumpq * xopt;
      ! Xpt.colwise() -= xopt;

      ! costruzione W e centratura di xpt
      W0 = matmul(m_xpt, m_pq) - m_half*sumpq*m_x_opt
      m_xpt = m_xpt - spread(m_x_opt, 2, m_npt)

      ! CAMBIATO: aggiornamento m_HQ (matrice piena upper triangular)
      do j = 1, m_nv
         do i = 1, j
            m_HQ(i, j) = m_HQ(i, j) + W0(i)*m_x_opt(j) + m_x_opt(i)*W0(j)
            m_HQ(j, i) = m_HQ(i, j)
         end do
         B1(j, 1:j) = B1(1:j, j)
      end do

      m_x_base = m_x_base + m_x_opt
      m_x_new = m_x_new - m_x_opt
      m_s_lower = m_s_lower - m_x_opt
      m_s_upper = m_s_upper - m_x_opt
      m_x_opt = m_zero
      m_x_opt_square = m_zero
   end subroutine model_shift_base

   subroutine update_optimal_point()
      implicit none

      real(wp) :: scaden, biglsq, densav, delsq
      real(wp) :: hdiag, den, distsq, temp
      integer  :: ksav, k

      delsq  = m_delta**2
      scaden = 0.0
      biglsq = 0.0
      ksav   = 0
      do k = 1, m_npt
         hdiag  = sum(m_Z(k, :)**2)
         den    = m_beta*hdiag + m_v_lag(k)**2
         distsq = sum((m_xpt(:, k) - m_x_new(:))**2)
         temp   = max(1.0, (distsq/delsq)**2)
         if ( temp*den > scaden) then
            scaden = temp*den
            ksav   = k
            densav = den
         end if
         biglsq = max( biglsq, temp * m_v_lag(k)**2 )
      end do
      if (scaden <= m_half*biglsq) then
         m_knew  = ksav
         m_denom = densav
      end if
   end subroutine update_optimal_point

   !*****************************************************************************************

   !*****************************************************************************************
   !>
   !  This subroutine seeks the least value of a function of many variables,
   !  by applying a trust region method that forms quadratic models by
   !  interpolation. There is usually some freedom in the interpolation
   !  conditions, which is taken up by minimizing the Frobenius norm of
   !  the change to the second derivative of the model, beginning with the
   !  zero matrix. The values of the variables are constrained by upper and
   !  lower bounds.
   !
   !  In addition to providing CALFUN, an initial vector of variables and
   !  the lower and upper bounds, the user has to set the values of the parameters
   !  ```RHOBEG```, ```RHOEND``` and ```NPT```. After scaling the individual variables
   !  if necessary, so that the magnitudes of their expected changes are similar,
   !  ```RHOBEG``` is the initial steplength for changes to the variables, a reasonable choice
   !  being the mesh size of a coarse grid search. Further, ```RHOEND``` should be suitable for
   !  a search on a very fine grid. Typically, the software calculates a vector
   !  of variables that is within distance ```10*RHOEND``` of a local minimum. Another
   !  consideration is that every trial vector of variables is forced to satisfy
   !  the lower and upper bounds, but there has to be room to make a search in all
   !  directions. Therefore an error return occurs if the difference between the
   !  bounds on any variable is less than ```2*RHOBEG```. The parameter ```NPT``` specifies
   !  the number of interpolation conditions on each quadratic model, the value
   !  ```NPT=2*N+1``` being recommended for a start, where ```N``` is the number of
   !  variables. It is often worthwhile to try other choices too, but much larger values
   !  tend to be inefficient, because the amount of routine work of each iteration is
   !  of magnitude ```NPT**2```, and because the achievement of adequate accuracy in some
   !  matrix calculations becomes more difficult. Some excellent numerical results
   !  have been found in the case ```NPT=N+6``` even with more than 100 variables.

   subroutine bobyqa(n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun, calfun)

      implicit none

      integer, intent(in) :: n       !! number of variables (must be at least two)
      integer, intent(in) :: npt     !! number of interpolation conditions. Its value must be in
      !! the interval [N+2,(N+1)(N+2)/2]. Choices that exceed 2*N+1 are not
      !! recommended.
      real(wp), dimension(:), intent(inout) :: x       !! Initial values of the variables must be set in X(1),X(2),...,X(N). They
      !! will be changed to the values that give the least calculated F.
      real(wp), dimension(:), intent(in) :: xl      !! lower bounds on x. The construction of quadratic models
      !! requires XL(I) to be strictly less than XU(I) for each I. Further,
      !! the contribution to a model from changes to the I-th variable is
      !! damaged severely by rounding errors if XU(I)-XL(I) is too small.
      real(wp), dimension(:), intent(in) :: xu      !! upper bounds on x. The construction of quadratic models
      !! requires XL(I) to be strictly less than XU(I) for each I. Further,
      !! the contribution to a model from changes to the I-th variable is
      !! damaged severely by rounding errors if XU(I)-XL(I) is too small.
      real(wp), intent(in) :: rhobeg  !! RHOBEG must be set to the initial value of a trust region radius.
      !! It must be positive, and typically should be about one tenth of the greatest
      !! expected change to a variable.  An error return occurs if any of
      !! the differences XU(I)-XL(I), I=1,...,N, is less than 2*RHOBEG.
      real(wp), intent(in) :: rhoend  !! RHOEND must be set to the final value of a trust
      !! region radius. It must be positive with RHOEND no greater than
      !! RHOBEG. Typically, RHOEND should indicate the
      !! accuracy that is required in the final values of the variables.
      integer, intent(in) :: iprint  !! IPRINT should be set to 0, 1, 2 or 3, which controls the
      !! amount of printing. Specifically, there is no output if IPRINT=0 and
      !! there is output only at the return if IPRINT=1. Otherwise, each new
      !! value of RHO is printed, with the best vector of variables so far and
      !! the corresponding value of the objective function. Further, each new
      !! value of F with its variables are output if IPRINT=3.
      integer, intent(in) :: maxfun  !! an upper bound on the number of calls of CALFUN.
      procedure(func) :: calfun  !! SUBROUTINE CALFUN (N,X,F) has to be provided by the user. It must set
      !! F to the value of the objective function for the current values of the
      !! variables X(1),X(2),...,X(N), which are generated automatically in a
      !! way that satisfies the bounds given in XL and XU.

      integer ::  i, j
      character(len=20) :: istate

      istate = 'INIT'
      m_nf   = 0

      do

         select case (istate)

         case ('INIT')

            write (*, *)
            write (*, *) 'INIT'
               
      write (*, *) 'initialize X is:', (x(i), i=1, m_nv)


            istate = initialize()
            cycle

         case ('UPDATE_GRADIENT')

            call info('UPDATE_GRADIENT')
            istate = update_gopt()
            cycle

         case ('TRUST_REGION')

            call info('TRUST_REGION')
            istate = trust_region_step()
            cycle

         case ('SHIFT_BASE')
            !
            !     Severe cancellation is likely to occur if XOPT is too far from XBASE.
            !     If the following test holds, then XBASE is shifted so that XOPT becomes
            !     zero. The appropriate changes are made to BMAT and to the second
            !     derivatives of the current model, beginning with the changes to BMAT
            !     that do not depend on ZMAT. VLAG is used temporarily for working space.
            !
            call info('SHIFT_BASE')
            !
            !     Generate the next point in the trust region that provides a small value
            !     of the quadratic model subject to the constraints on the variables.
            !     The integer m_ntrits is set to the number "trust region" iterations that
            !     have occurred since the last "alternative" iteration. If the length
            !     of XNEW-XOPT is less than m_half*RHO, however, then there is a branch to
            !     label 650 or 680 with m_ntrits=-1, instead of calculating F at XNEW.
            !
            if (m_dsq <= 1.0e-3_wp*m_x_opt_square) then
               call model_shift_base()
            end if

            if (m_ntrits == 0) then
               istate = 'ALTMOV'
               cycle
            end if

            istate = 'COMPUTE_VLAG'
            cycle

         case ('RESCUE')
            !
            !     XBASE is also moved to XOPT by a call of RESCUE. This calculation is
            !     more expensive than the previous shift, because new matrices BMAT and
            !     ZMAT are generated from scratch, which may include the replacement of
            !     interpolation points whose positions seem to be causing near linear
            !     dependence in the interpolation conditions. Therefore RESCUE is called
            !     only if rounding errors have reduced by at least a factor of two the
            !     denominator of the formula for updating the H matrix. It provides a
            !     useful safeguard, but is not invoked in most applications of BOBYQA.
            !
            call info('RESCUE')
            istate = rescue_step()
            cycle

         case ('ALTMOV')
            !
            !     Pick two alternative vectors of variables, relative to XBASE, that
            !     are suitable as new positions of the KNEW-th interpolation point.
            !     Firstly, XNEW is set to the point on a line through XOPT and another
            !     interpolation point that minimizes the predicted value of the next
            !     denominator, subject to ||XNEW - XOPT|| .LEQ. ADELT and to the SL
            !     and SU bounds. Secondly, XALT is set to the best feasible point on
            !     a constrained version of the Cauchy step of the KNEW-th Lagrange
            !     function, the corresponding value of the square of this function
            !     being returned in CAUCHY. The choice between these alternatives is
            !     going to be made when the denominator is calculated.
            !
            call info('ALTMOV')
            call altmov()

            istate = 'COMPUTE_VLAG'
            cycle

         case ('COMPUTE_VLAG')

            call info('COMPUTE_VLAG')
            istate = compute_vlag()
            cycle

         case ('EVALUATE')

            call info('EVALUATE')
            istate = add_new_point()
            cycle

         case ('FIND_FAR')

            call info('FIND_FAR')
            istate = find_far()
            cycle

         case ('REDUCE_RHO')

            call info('REDUCE_RHO')
            istate = reduce_rho()
            cycle

         case ('DONE')
            call info('DONE')
            exit

         end select
      end do

      !
      !     Return from the calculation, after another Newton-Raphson step, if
      !       it is too short to have been tried before.
      !
      if (m_f_val(m_kopt) <= m_fsave) then
         ! clipping base + opt tra lower e upper
         x = min(max(m_x_lower, m_x_base + m_x_opt), m_x_upper)
         ! gestire casi speciali NON SERVE
         ! x = merge(m_x_lower, x, m_x_opt == m_s_lower)
         ! x = merge(m_x_upper, x, m_x_opt == m_s_upper)
         !f = m_f_val(m_kopt)
      end if
      if (m_iprint >= 1) then
         write (*, '(A)') 'At the return from BOBYQA'
         write (*, '(A,I6)') 'Number of function values =', m_nf
         write (*, '(A,G15.8)') 'Least value of F =', m_f_val(m_kopt)
         write (*, '(A,*(G15.8))') 'The corresponding X is:', (x(i), i=1, min(n,5))
      end if

      call free_memory
      return

   contains

      function initialize()

         implicit none

         character(len=20) :: initialize

         m_nv   = n
         m_npt  = npt
         m_ndim = npt + n
         m_nptm = npt - n - 1

         allocate (m_xpt(n, npt))
         allocate (m_B(n, npt + n))
         allocate (m_Z(npt, m_nptm))
         allocate (m_x_base(n))
         allocate (m_x_new(n))
         allocate (m_g_new(n))
         allocate (m_x_opt(n))
         allocate (m_g_opt(n))
         allocate (m_g_lag(n))
         allocate (m_x_alt(n))
         allocate (m_f_val(npt))
         allocate (m_v_lag(npt + n))
         allocate (m_pq(npt))
         allocate (m_HQ(n, n))  ! CAMBIATO: matrice piena n x n

         allocate (m_s_lower(n))
         allocate (m_s_upper(n))
         allocate (m_x_lower(n))
         allocate (m_x_upper(n))

         allocate (m_xbdi(n))
         allocate (m_s(n))
         allocate (m_hs(n))
         allocate (m_hred(n))

         m_rhobeg = rhobeg
         m_rhoend = rhoend
         m_maxfun = maxfun
         m_iprint = iprint

         !
         !     Return if the value of NPT is unacceptable.
         !
         if (m_npt < m_nv + 2 .or. m_npt > ((m_nv + 2)*(m_nv + 1))/2) then
            write (*, *) 'Return from BOBYQA because NPT is not in the required interval'
            return
         end if

         m_x_lower = xl
         m_x_upper = xu

         !
         !     Return if there is insufficient space between the bounds. Modify the
         !     initial X if necessary in order to avoid conflicts between the bounds
         !     and the construction of the first quadratic model. The lower and upper
         !     bounds on moves from the updated X are set now, in the ISL and ISU
         !     partitions of W, in order to provide useful and exact information about
         !     components of X that become within distance RHOBEG from their bounds.
         !

         if (any(m_x_upper - m_x_lower < 2.0_wp*m_rhobeg)) then
            write (*, *) 'Return from BOBYQA because one of the differences...'
            initialize = 'DONE'
            return
         end if

         call box_adjust(x)

         !
         !     The call of PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
         !     BMAT and ZMAT for the first iteration, with the corresponding values of
         !     of NF and KOPT, which are the number of calls of CALFUN so far and the
         !     index of the interpolation point at the trust region centre. Then the
         !     initial XOPT is set too. The branch to label 720 occurs if MAXFUN is
         !     less than NPT. GOPT will be updated if KOPT is different from KBASE.
         !

         call prelim(x, calfun)

         m_x_opt = m_xpt(:, m_kopt)
         m_x_opt_square = squaredNorm(m_x_opt)
         m_fsave = m_f_val(1)
         if (m_nf < m_npt) then
            if (m_iprint > 0) write (*, *) 'Return from BOBYQA because CALFUN has been called MAXFUN times.'
            initialize = 'DONE'
            return
         end if
         m_kbase = 1
         !
         !     Complete the settings that are required for the iterative procedure.
         !
         m_rho = m_rhobeg
         m_delta = m_rho
         m_nresc = m_nf
         m_ntrits = 0
         m_diffa = m_zero
         m_diffb = m_zero
         m_itest = 0
         m_nf_saved = m_nf

         initialize = 'UPDATE_GRADIENT'
         return
      end function initialize

      subroutine free_memory()

         if (allocated(m_xpt)) deallocate (m_xpt)
         if (allocated(m_B)) deallocate (m_B)
         if (allocated(m_Z)) deallocate (m_Z)
         if (allocated(m_x_base)) deallocate (m_x_base)
         if (allocated(m_x_new)) deallocate (m_x_new)
         if (allocated(m_g_new)) deallocate (m_g_new)
         if (allocated(m_x_opt)) deallocate (m_x_opt)
         if (allocated(m_g_opt)) deallocate (m_g_opt)
         if (allocated(m_g_lag)) deallocate (m_g_lag)
         if (allocated(m_x_alt)) deallocate (m_x_alt)
         if (allocated(m_f_val)) deallocate (m_f_val)
         if (allocated(m_v_lag)) deallocate (m_v_lag)
         if (allocated(m_pq)) deallocate (m_pq)
         if (allocated(m_HQ)) deallocate (m_HQ)  ! matrice piena n x n

         if (allocated(m_s_lower)) deallocate (m_s_lower)
         if (allocated(m_s_upper)) deallocate (m_s_upper)
         if (allocated(m_x_lower)) deallocate (m_x_lower)
         if (allocated(m_x_upper)) deallocate (m_x_upper)

         if (allocated(m_xbdi)) deallocate (m_xbdi)
         if (allocated(m_s)) deallocate (m_s)
         if (allocated(m_hs)) deallocate (m_hs)
         if (allocated(m_hred)) deallocate (m_hred)

      end subroutine free_memory

      !
      !     Update GOPT if necessary before the first iteration and after each
      !     call of RESCUE that makes a call of CALFUN.
      !
      function update_gopt()

         implicit none

         character(len=20) :: update_gopt

         if (m_kopt /= m_kbase) then
            m_g_opt = m_g_opt + grad_times_vector(m_x_opt)
         end if

         update_gopt = 'TRUST_REGION'
         return

      end function update_gopt

      function trust_region_step()

         implicit none

         character(len=20) :: trust_region_step

         call trsbox()

         m_dnorm = min(m_delta, sqrt(m_dsq))
         if (m_dnorm < m_half*m_rho) then
            m_ntrits = -1
            m_distsq = (m_ten*m_rho)**2
            if (m_nf <= m_nf_saved + 2) then
               trust_region_step = 'FIND_FAR'
               return
            end if
            !
            !     The following choice between labels 650 and 680 depends on whether or
            !     not our work with the current RHO seems to be complete. Either RHO is
            !     decreased or termination occurs if the errors in the quadratic model at
            !     the last three interpolation points compare favourably with predictions
            !     of likely improvements to the model within distance m_half*RHO of XOPT.
            !
            block
               real(wp) :: errbig, frhosq, bdtol, bdtest, curv
               errbig = max(m_diffa, m_diffb, m_diffc)
               frhosq = 0.125_wp*m_rho*m_rho
               if (m_crvmin > m_zero .and. errbig > frhosq*m_crvmin) then
                  trust_region_step = 'FIND_FAR'
                  return
               end if

               bdtol = errbig/m_rho
               do j = 1, n
                  bdtest = bdtol
                  if (m_x_new(j) == m_s_lower(j)) bdtest = m_g_new(j)
                  if (m_x_new(j) == m_s_upper(j)) bdtest = -m_g_new(j)
                  if (bdtest < bdtol) then
                     curv = m_HQ(j, j) + sum(m_pq*m_xpt(j, :)**2)
                     bdtest = bdtest + m_half*curv*m_rho
                     if (bdtest < bdtol) then
                        trust_region_step = 'FIND_FAR'
                        return
                     end if
                  end if
               end do
            end block
            trust_region_step = 'REDUCE_RHO'
            return
         end if
         m_ntrits = m_ntrits + 1

         trust_region_step = 'SHIFT_BASE'
         return
      end function trust_region_step


      function rescue_step()

         implicit none

         character(len=20) :: rescue_step

         !
         !     XBASE is also moved to XOPT by a call of RESCUE. This calculation is
         !     more expensive than the previous shift, because new matrices BMAT and
         !     ZMAT are generated from scratch, which may include the replacement of
         !     interpolation points whose positions seem to be causing near linear
         !     dependence in the interpolation conditions. Therefore RESCUE is called
         !     only if rounding errors have reduced by at least a factor of two the
         !     denominator of the formula for updating the H matrix. It provides a
         !     useful safeguard, but is not invoked in most applications of BOBYQA.
         !
         m_nf_saved = m_nf
         m_kbase = m_kopt
         call rescue(n, calfun)
         !
         !     XOPT is updated now in case the branch below to label 720 is taken.
         !     Any updating of GOPT occurs after the branch below to label 20, which
         !     leads to a trust region iteration as does the branch to label 60.
         !
         m_x_opt_square = m_zero
         if (m_kopt /= m_kbase) then
            m_x_opt = m_xpt(:, m_kopt)
            m_x_opt_square = sum(m_x_opt**2)
         end if
         if (m_nf < 0) then
            m_nf = m_maxfun
            if (m_iprint > 0) write (*, *) 'Return from BOBYQA because CALFUN has been called MAXFUN times.'
            rescue_step = 'DONE'
            return
         end if
         m_nresc = m_nf
         if (m_nf_saved < m_nf) then
            m_nf_saved = m_nf
            rescue_step = 'UPDATE_GRADIENT'
            return
         end if
         if (m_ntrits > 0) then
            rescue_step = 'TRUST_REGION'
            return
         end if

         rescue_step = 'ALTMOV'
      end function rescue_step

      function compute_vlag() result(next_action)
         implicit none
         character(len=20) :: next_action

         ! Variabili locali
         real(wp) :: W(m_npt), a_vec(m_npt), b_vec(m_npt), proj(m_nptm), d(m_nv), dx, bsum
         real(wp) :: temp, hdiag, den, scaden, biglsq, delsq

         ! --- Dichiarazione Alias (Puntatori) ---
         ! Rimuovi 'allocatable, target' e usa solo 'pointer'
         real(wp), pointer :: v_lag0(:), v_lag1(:)
         real(wp), pointer :: B0(:, :), B1(:, :)
         integer :: k

         ! --- Associazione Alias ---
         v_lag0 => m_v_lag(1:m_npt)
         v_lag1 => m_v_lag(m_npt + 1:m_npt + m_nv)
         B0 => m_B(1:m_nv, 1:m_npt)
         B1 => m_B(1:m_nv, m_npt + 1:m_npt + m_nv)

         d = m_x_new - m_x_opt

         ! Calcola separatamente le proiezioni
         a_vec = matmul(transpose(m_xpt), d)     ! XPTᵀ·d (npt elementi)
         b_vec = matmul(transpose(m_xpt), m_x_opt) ! XPTᵀ·x_opt (npt elementi)
         W = a_vec*(m_half*a_vec + b_vec)

         ! =====================================================================
         ! CALCOLO β E AGGIORNAMENTO VLAG CON MATRICE Z
         ! =====================================================================
         ! β = -Σ_{j=1}^{nptm} (Z_j^T·W)^2 (curvatura negativa da parte a rango ridotto)
         ! v_lag ← v_lag + Σ_{j=1}^{nptm} (Z_j^T·W)·Z_j
         proj = matmul(transpose(m_Z), W)
         v_lag0 = matmul(transpose(B0), d) + matmul(m_Z, proj) ! v_lag0 = B_k^T·d + Z·proj
         m_beta = -dot_product(proj, proj)                     ! -‖proj‖²

         ! =====================================================================
         ! CALCOLO NORMA E PRODOTTI SCALARI AUSILIARI
         ! =====================================================================
         ! ||d||^2 e d^T·x_opt
         m_dsq = dot_product(d, d)
         dx = dot_product(d, m_x_opt)

         ! =====================================================================
         ! 5. CALCOLO PARTE ESTESA DI VLAG (COMPONENTI n+1...n+npt)
         ! =====================================================================
         ! v_{npt+1:npt+n} = B_{1:n,1:npt}·W + B_{1:n,npt+1:npt+n}^T·d
         ! bsum = 2·d^T·(B_{1:n,1:npt}·W + ½·B_{1:n,npt+1:npt+n}^T·d)
         v_lag1 = matmul(B0, W)
         bsum = dot_product(d, v_lag1)

         v_lag1 = v_lag1 + matmul(transpose(B1), d)
         bsum = bsum + dot_product(d, v_lag1)

         ! =====================================================================
         ! CALCOLO FINALE DI β
         ! =====================================================================
         ! β = (d^T·x_opt)^2 + ||d||^2·(||x_opt||^2 + 2·d^T·x_opt + ½·||d||^2)
         !     - Σ(Z_j^T·W)^2 - 2·d^T·(B·W + B^T·d)
         m_beta = dx*dx + m_dsq*(m_x_opt_square + 2*dx + m_half*m_dsq) + m_beta - bsum

         ! =====================================================================
         ! CONDIZIONE DI INTERPOLAZIONE PER PUNTO OTTIMO
         ! =====================================================================
         ! L_kopt(x_opt) = δ_{k,kopt} → v_kopt ← v_kopt + 1
         m_v_lag(m_kopt) = m_v_lag(m_kopt) + m_one

         ! =====================================================================
         ! 8. GESTIONE CASO m_ntrits = 0 (PRIMO TENTATIVO O RESTART)
         ! =====================================================================
         if (m_ntrits == 0) then
            ! -------------------------------------------------------------
            ! 8.1 CALCOLO DENOMINATORE: denom = v_knew^2 + α·β
            ! -------------------------------------------------------------
            m_denom = m_v_lag(m_knew)**2 + m_alpha*m_beta

            ! -------------------------------------------------------------
            ! 8.2 CONTROLLO PASSO DI CAUCHY
            ! -------------------------------------------------------------
            ! Se denom < m_cauchy, usa passo di Cauchy
            if ( m_denom < m_cauchy .and. m_cauchy > m_zero ) then
               m_x_new = m_x_alt
               m_cauchy = m_zero
               next_action = 'COMPUTE_VLAG'  ! Ricalcola con passo di Cauchy
               return
            end if

            ! -------------------------------------------------------------
            ! 8.3 CONTROLLO CANCELLAZIONE NUMERICA
            ! -------------------------------------------------------------
            if (m_denom <= m_half*m_v_lag(m_knew)**2) then
               if (m_nf > m_nresc) then
                  next_action = 'RESCUE'  ! Troppa cancellazione, chiama rescue
                  return
               end if
               if (m_iprint > 0) then
                  write (*, *) 'Return from BOBYQA because of much cancellation in a denominator.'
               end if
               next_action = 'DONE'
               return
            end if

            ! =====================================================================
            ! 9. CASO m_ntrits > 0 (ITERAZIONI SUCCESSIVE)
            ! =====================================================================
         else
            ! -------------------------------------------------------------
            ! 9.1 INIZIALIZZAZIONE PER SELEZIONE PUNTO DA ELIMINARE
            ! -------------------------------------------------------------
            delsq  = m_delta**2
            scaden = 0.0_wp
            biglsq = 0.0_wp
            m_knew = 0

            ! -------------------------------------------------------------
            ! 9.2 RICERCA PUNTO CON MASSIMO SCADEN
            ! -------------------------------------------------------------
            ! SCADEN = max_{k≠kopt} [max(1, (||x_k-x_opt||^2/Δ^2)^2]·[β·||Z_k||^2 + v_k^2]
            do k = 1, m_npt
               if (k == m_kopt) cycle

               ! Norma della k-esima riga di Z: hdiag = ||Z_k||^2
               hdiag = dot_product(m_Z(k, :), m_Z(k, :))

               ! Denominatore: den = β·hdiag + v_k^2
               den = m_beta*hdiag + m_v_lag(k)**2

               ! Distanza normalizzata: temp = max(1, (distsq/Δ^2)^2)
               m_distsq = dot_product(m_xpt(:, k) - m_x_opt, m_xpt(:, k) - m_x_opt)
               temp = max(1.0_wp, (m_distsq/delsq)**2)

               ! Aggiorna massimo SCADEN
               if (temp*den > scaden) then
                  scaden = temp*den
                  m_knew = k
                  m_denom = den
               end if

               ! Aggiorna massimo BIGLSQ
               biglsq = max(biglsq, temp*m_v_lag(k)**2)
            end do

            ! -------------------------------------------------------------
            ! 9.3 CONTROLLO CANCELLAZIONE NUMERICA
            ! -------------------------------------------------------------
            if (scaden <= m_half*biglsq) then
               if (m_nf > m_nresc) then
                  next_action = 'RESCUE'
                  return
               end if
               if (m_iprint > 0) then
                  write (*, *) 'Return from BOBYQA because of much cancellation in a denominator.'
               end if
               next_action = 'DONE'
               return
            end if
         end if

         ! =====================================================================
         ! 10. USCITA NORMALE: VALUTA FUNZIONE OBIETTIVO
         ! =====================================================================
         next_action = 'EVALUATE'

      end function compute_vlag

      function add_new_point()
         implicit none

         character(len=20) :: add_new_point

         real(wp) :: f, vquad, diff, fopt
         real(wp) :: W1(m_npt), d(m_nv), proj(m_nptm)

         !
         !     Put the variables for the next calculation of the objective function
         !       in XNEW, with any adjustments for the bounds.
         !
         !
         !     Calculate the value of the objective function at XBASE+XNEW, unless
         !       the limit on the number of calculations of F has been reached.
         !

         ! clipping base+new tra lower e upper
         x = m_x_base + m_x_new
         call box_project( x )

         d = m_x_new - m_x_opt

         if (m_nf >= m_maxfun) then
            if (m_iprint > 0) write (*, *) 'Return from BOBYQA because CALFUN has been called MAXFUN times.'
            add_new_point = 'DONE'
            return
         end if
         m_nf = m_nf + 1
         call calfun(n, x, f)
         if (m_iprint == 3) then
            write (*, '(A,I6,A,G15.8,A,*(G15.8))') 'Function n.', m_nf, ' F(X) = ', f, '   X: ', x(1:min(n,5))
         end if
         if (m_ntrits == -1) then
            m_fsave = f
            add_new_point = 'DONE'
            return
         end if
         !
         !     Use the quadratic model to predict the change in F due to the step D,
         !       and set DIFF to the error of this prediction.
         !
         fopt = m_f_val(m_kopt)
         vquad = dHd(d)

         diff = f - fopt - vquad
         m_diffc = m_diffb
         m_diffb = m_diffa
         m_diffa = abs(diff)
         if (m_dnorm > m_rho) m_nf_saved = m_nf
         !
         !     Pick the next value of DELTA after a trust region step.
         !
         if (m_ntrits > 0) then
            if (vquad >= m_zero) then
               if (m_iprint > 0) write (*, *) 'Return from BOBYQA because a trust region step has failed to reduce Q.'
               add_new_point = 'DONE'
               return
            end if
            m_ratio = (f - fopt)/vquad
            if (m_ratio <= m_tenth) then
               m_delta = min(m_half*m_delta, m_dnorm)
            else if (m_ratio <= 0.7_wp) then
               m_delta = max(m_half*m_delta, m_dnorm)
            else
               m_delta = max(m_half*m_delta, 2*m_dnorm)
            end if
            if (m_delta <= 1.5_wp*m_rho) m_delta = m_rho
            !
            !     Recalculate KNEW and DENOM if the new F is less than FOPT.
            !
            if (f < fopt) then
               call update_optimal_point()
            end if
         end if
         !
         !     Update BMAT and ZMAT, so that the KNEW-th interpolation point can be
         !     moved. Also update the second derivative terms of the model.
         !
         call update()

         block
            real(wp) :: pqold
            pqold = m_pq(m_knew)
            m_pq(m_knew) = 0.0

            ! CAMBIATO: aggiornamento m_HQ (matrice piena upper triangular)
            do i = 1, n
               do j = 1, i
                  m_HQ(j, i) = m_HQ(j, i) + pqold*m_xpt(i, m_knew)*m_xpt(j, m_knew)
                  m_HQ(i, j) = m_HQ(j, i)
               end do
            end do
         end block

         ! Aggiornamento m_pq
         m_pq = m_pq + diff*matmul(m_Z, m_Z(m_knew, :))

         !
         !     Include the new interpolation point, and make the changes to GOPT at
         !     the old XOPT that are caused by the updating of the quadratic model.
         !
         m_f_val(m_knew) = f
         m_xpt(:, m_knew) = m_x_new

         ! Z·Z(knew,:)ᵀ + XPTᵀ·x_opt
         W1 = matmul(m_Z, m_Z(m_knew, :))*matmul(transpose(m_xpt), m_x_opt)
         m_g_opt = m_g_opt + diff*(m_B(:, m_knew) + matmul(m_xpt, W1))

         !
         !     Update XOPT, GOPT and KOPT if the new calculated F is less than FOPT.
         !
         if (f < fopt) then
            m_kopt = m_knew
            m_x_opt = m_x_new
            m_x_opt_square = squaredNorm(m_x_opt)
            !call add_Hv(d, m_g_opt)
            m_g_opt = m_g_opt + grad_times_vector(d)
         end if
         !
         !     Calculate the parameters of the least Frobenius norm interpolant to
         !     the current data, the gradient of this interpolant at XOPT being put
         !     into VLAG(NPT+I), I=1,2,...,N.
         !
         if (m_ntrits > 0) then

            block
               real(wp) :: V0(m_npt), V1(m_nv), W2(m_npt), gqsq, gisq, temp1(m_nv), temp2(m_nv)
               V0 = m_f_val - m_f_val(m_kopt)

               ! Passo 1: Calcola proiezioni c_j = Z(:,j)·v_lag
               proj = matmul(transpose(m_Z), V0)
               W2 = matmul(m_Z, proj)

               ! Passo 2: Ricostruisce W1 = Σ_j c_j * Z(:,j)
               W1 = matmul(transpose(m_xpt), m_x_opt) * W2
    
               V1 = matmul(m_B(:,1:m_npt), V0) + matmul(m_xpt, W1)

               temp1 = V1;
               temp2 = m_g_opt;
               where (abs(m_x_opt - m_s_lower) <= m_eps)
                  temp1 = min(0.0_wp, temp1)
                  temp2 = min(0.0_wp, temp2)
               else where (abs(m_x_opt - m_s_upper) <= m_eps)
                  temp1 = max(0.0_wp, temp1)
                  temp2 = max(0.0_wp, temp2)
               end where
               gqsq = squaredNorm( temp2 )
               gisq = squaredNorm( temp1 )
               !
               !     Test whether to replace the new quadratic model by the least Frobenius
               !     norm interpolant, making the replacement if the test is satisfied.
               !
               m_itest = m_itest + 1
               if (gqsq < m_ten*gisq) m_itest = 0
               if (m_itest >= 3) then
                  m_g_opt = V1
                  m_pq    = W2
                  m_HQ    = 0
               end if
            end block
         end if
         !
         !     If a trust region step has provided a sufficient decrease in F, then
         !     branch for another trust region calculation. The case m_ntrits=0 occurs
         !     when the new interpolation point was reached by an alternative step.
         !
         if (m_ntrits == 0) then
            add_new_point = 'TRUST_REGION'
            return
         end if
         if (f <= fopt + m_tenth*vquad) then
            add_new_point = 'TRUST_REGION'
            return
         end if
         !
         !     Alternatively, find out if the interpolation points are close enough
         !       to the best point so far.
         !
         m_distsq = max((m_two*m_delta)**2, (m_ten*m_rho)**2)

         add_new_point = 'FIND_FAR'
         return
      end function add_new_point

      function find_far()
         implicit none

         character(len=20) :: find_far

         real(wp) :: a_vec(m_npt)
         real(wp) :: dist, xopt_norm2

         xopt_norm2 = squaredNorm(m_x_opt)
         a_vec = sum(m_xpt**2, dim=1) - 2.0_wp*matmul(m_x_opt, m_xpt) + xopt_norm2

         ! Trova massimo valore e indice
         m_knew = maxloc(a_vec, dim=1)  ! Indice del massimo
         if (a_vec(m_knew) > m_distsq) then
            m_distsq = a_vec(m_knew)
            !
            !     If KNEW is positive, then ALTMOV finds alternative new positions for
            !     the KNEW-th interpolation point within distance ADELT of XOPT. It is
            !     reached via label 90. Otherwise, there is a branch to label 60 for
            !     another trust region iteration, unless the calculations with the
            !     current RHO are complete.
            !
            dist = sqrt(m_distsq)
            if (m_ntrits == -1) then
               m_delta = min(m_tenth*m_delta, m_half*dist)
               if (m_delta <= 1.5_wp*m_rho) m_delta = m_rho
            end if
            m_ntrits = 0
            m_adelt = max(min(m_tenth*dist, m_delta), m_rho)
            m_dsq = m_adelt*m_adelt
            find_far = 'SHIFT_BASE'
            return
         else
            m_knew = 0
         end if
         if (m_ntrits == -1) then
            find_far = 'REDUCE_RHO'
            return
         end if
         if (m_ratio > m_zero) then
            find_far = 'TRUST_REGION'
            return
         end if
         if (max(m_delta, m_dnorm) > m_rho) then
            find_far = 'TRUST_REGION'
            return
         end if
         find_far = 'REDUCE_RHO'
         return
      end function find_far

      function reduce_rho()
         implicit none

         character(len=20) :: reduce_rho
         !
         !     The calculations with the current value of RHO are complete. Pick the
         !       next values of RHO and DELTA.
         !
         if (m_rho > m_rhoend) then
            m_delta = m_half*m_rho
            m_ratio = m_rho/m_rhoend
            if (m_ratio <= 16.0_wp) then
               m_rho = m_rhoend
            else if (m_ratio <= 250.0_wp) then
               m_rho = sqrt(m_ratio)*m_rhoend
            else
               m_rho = m_tenth*m_rho
            end if
            m_delta = max(m_delta, m_rho)
            if (m_iprint >= 2) then
               if (m_iprint >= 3) write (*, '(5x)')
               write (*, '(A,G15.8,1X,A,I6)') 'New RHO =', m_rho, 'Number of function values =', m_nf
               write (*, '(A,G15.8)') 'Least value of F =', m_f_val(m_kopt)
               write (*, '(A,*(G15.8))') 'The corresponding X(b+o) is:', (m_x_base(i) + m_x_opt(i), i=1, min(n,5))
            end if
            m_ntrits = 0
            m_nf_saved = m_nf
            reduce_rho = 'TRUST_REGION'
            return
         end if

         if (m_ntrits == -1) then
            reduce_rho = 'EVALUATE'
            return
         end if
         reduce_rho = 'DONE'
         return
      end function reduce_rho

   end subroutine bobyqa

   subroutine altmov()

      implicit none

      real(wp) :: ha, presav, bigstp, curv, scale, ggfree, gw, step, stpsav, temp, wfixsq, wsqsav
      real(wp) :: W(m_nv), hcol(m_npt)
      real(wp) :: x_alt_saved(m_nv), cauchy_saved

      integer :: ibdsav
      integer :: i, k, i_target, iflag, istate, ksav

      ! --- Dichiarazione Alias (Puntatori) ---
      ! Rimuovi 'allocatable, target' e usa solo 'pointer'
      real(wp), pointer :: v_lag0(:), v_lag1(:)
      real(wp), pointer :: B0(:, :), B1(:, :)

      ! --- Associazione Alias ---
      v_lag0 => m_v_lag(1:m_npt)
      v_lag1 => m_v_lag(m_npt + 1:m_npt + m_nv)
      B0 => m_B(1:m_nv, 1:m_npt)
      B1 => m_B(1:m_nv, m_npt + 1:m_npt + m_nv)

      !
      !     The arguments N, NPT, XPT, XOPT, BMAT, ZMAT, NDIM, SL and SU all have
      !       the same meanings as the corresponding arguments of BOBYQB.
      !     KOPT is the index of the optimal interpolation point.
      !     KNEW is the index of the interpolation point that is going to be moved.
      !     ADELT is the current trust region bound.
      !     XNEW will be set to a suitable new position for the interpolation point
      !       XPT(KNEW,.). Specifically, it satisfies the SL, SU and trust region
      !       bounds and it should provide a large denominator in the next call of
      !       UPDATE. The step XNEW-XOPT from XOPT is restricted to moves along the
      !       straight lines through XOPT and another interpolation point.
      !     XALT also provides a large value of the modulus of the KNEW-th Lagrange
      !       function subject to the constraints that have been mentioned, its main
      !       difference from XNEW being that XALT-XOPT is a constrained version of
      !       the Cauchy step within the trust region. An exception is that XALT is
      !       not calculated if all components of GLAG (see below) are zero.
      !     ALPHA will be set to the KNEW-th diagonal element of the H matrix.
      !     CAUCHY will be set to the square of the KNEW-th Lagrange function at
      !       the step XALT-XOPT from XOPT for the vector XALT that is returned,
      !       except that CAUCHY is set to zero if XALT is not calculated.
      !     GLAG is a working space vector of length N for the gradient of the
      !       KNEW-th Lagrange function at XOPT.
      !     HCOL is a working space vector of length NPT for the second derivative
      !       coefficients of the KNEW-th Lagrange function.
      !     W is a working space vector of length 2N that is going to hold the
      !       constrained Cauchy step from XOPT of the Lagrange function, followed
      !       by the downhill version of XALT when the uphill step is calculated.
      !
      !     Set the first NPT components of W to the leading elements of the
      !     KNEW-th column of the H matrix.
      !
      hcol    = matmul(m_Z, m_Z(m_knew, :))
      m_alpha = hcol(m_knew)
      ha      = m_half*m_alpha
      !
      !     Calculate the gradient of the KNEW-th Lagrange function at XOPT.
      !
      m_g_lag = m_B(:, m_knew) + matmul(m_xpt, matmul(transpose(m_xpt), m_x_opt)*hcol)

      !
      !     Search for a large denominator along the straight lines through XOPT
      !     and another interpolation point. SLBD and SUBD will be lower and upper
      !     bounds on the step along each of these lines in turn. PREDSQ will be
      !     set to the square of the predicted denominator for each line. PRESAV
      !     will be set to the largest admissible value of PREDSQ that occurs.
      !
      presav = m_zero
      ksav   = 0
      do k = 1, m_npt
         if (k == m_kopt) cycle
         block
            real(wp) :: ratio_sl(m_nv), ratio_su(m_nv), vdiff(m_nv)
            real(wp) :: slbd, subd, dderiv, diff, predsq, tempa, tempb, tempd, vlag
            integer  :: ilbd, iubd, isbd

            vdiff    = m_xpt(:, k) - m_x_opt
            dderiv   = dot_product(m_g_lag, vdiff)
            m_distsq = dot_product(vdiff, vdiff)

            subd = min(m_one, m_adelt/sqrt(m_distsq))
            slbd = -subd
            ilbd = 0
            iubd = 0

            where (vdiff > m_eps)
               ratio_sl = (m_s_lower - m_x_opt)/vdiff;
               ratio_su = (m_s_upper - m_x_opt)/vdiff;
            elsewhere(vdiff < -m_eps)
               ratio_sl = (m_s_upper - m_x_opt)/vdiff;
               ratio_su = (m_s_lower - m_x_opt)/vdiff;
            elsewhere
               ratio_sl = -huge(0.0_wp);
               ratio_su = huge(0.0_wp);
            end where

            ! 4. Aggiornamento di SLBD (Limite Inferiore dello scalare)
            ! Cerchiamo il rapporto più restrittivo (il valore massimo tra i rapporti negativi)
            if (any(ratio_sl > slbd)) then
               i_target = maxloc(ratio_sl, dim=1)
               slbd     = ratio_sl(i_target)
               ilbd     = merge(-i_target, i_target, vdiff(i_target) > 0 )
            end if

            ! 5. Aggiornamento di SUBD (Limite Superiore dello scalare)
            ! Cerchiamo il rapporto più restrittivo (il valore minimo tra i rapporti positivi)
            if (any(ratio_su < subd)) then
               i_target = minloc(ratio_su,dim=1)
               subd     = ratio_su(i_target)
               iubd     = merge(i_target, -i_target, vdiff(i_target) > 0 )
            end if

            !
            !        Seek a large modulus of the KNEW-th Lagrange function when the index
            !        of the other interpolation point on the line through XOPT is KNEW.
            !
            if (k == m_knew) then
               diff = dderiv - m_one
               step = slbd
               vlag = slbd*(dderiv - slbd*diff)
               isbd = ilbd
               temp = subd*(dderiv - subd*diff)
               if (abs(temp) > abs(vlag)) then
                  step = subd
                  vlag = temp
                  isbd = iubd
               end if
               tempd = m_half*dderiv
               tempa = tempd - diff*slbd
               tempb = tempd - diff*subd
               if (tempa*tempb < m_zero) then
                  temp = tempd*tempd/diff
                  if (abs(temp) > abs(vlag)) then
                     step = tempd/diff
                     vlag = temp
                     isbd = 0
                  end if
               end if
               !
               !     Search along each of the other lines through XOPT and another point.
               !
            else
               step = slbd
               vlag = slbd*(m_one - slbd)
               isbd = ilbd
               temp = subd*(m_one - subd)
               if (abs(temp) > abs(vlag)) then
                  step = subd
                  vlag = temp
                  isbd = iubd
               end if
               if (subd > m_half) then
                  if (abs(vlag) < 0.25_wp) then
                     step = m_half
                     vlag = 0.25_wp
                     isbd = 0
                  end if
               end if
               vlag = vlag*dderiv
            end if
            !
            !     Calculate PREDSQ for the current line search and maintain PRESAV.
            !
            temp = step*(m_one - step)*m_distsq
            predsq = vlag*vlag*(vlag*vlag + ha*temp*temp)
            if (predsq > presav .or. ksav .eq. 0 ) then
               presav = predsq
               ksav = k
               stpsav = step
               ibdsav = isbd
            end if

         end block
      end do
      !
      !     Construct XNEW in a way that satisfies the bound constraints exactly.
      !
      m_x_new = m_x_opt + stpsav*(m_xpt(:, ksav) - m_x_opt)
      call box_project_s( m_x_new )
      if (ibdsav < 0) m_x_new(-ibdsav) = m_s_lower(-ibdsav)
      if (ibdsav > 0) m_x_new(ibdsav) = m_s_upper(ibdsav)
      !
      !     Prepare for the iterative method that assembles the constrained Cauchy
      !     step in W. The sum of squares of the fixed components of W is formed in
      !     WFIXSQ, and the free components of W are set to BIGSTP.
      !
      bigstp = m_adelt + m_adelt
      iflag = 0

      istate = 100

      do

         select case (istate)

         case (100)

            ! inizializza
            W(1:m_nv) = 0.0
            wfixsq = 0.0
            block
               logical :: mask(m_nv)

               mask = (min(m_x_opt - m_s_lower, m_g_lag) > 0.0_wp) .or. (max(m_x_opt - m_s_upper, m_g_lag) < 0.0_wp)

               ! 3. Aggiornamento selettivo di W (Masked Assignment)
               where (mask) W = bigstp

               ! 4. Riduzione condizionale per ggfree
               ! La funzione sum accetta un argomento 'mask' opzionale
               ggfree = sum(m_g_lag**2, mask=mask)
            end block
            if (ggfree == m_zero) then
               m_cauchy = m_zero
               return
            end if

            istate = 120
            cycle

         case (120)
            !
            !     Investigate whether more components of W can be fixed.
            !
            temp = m_adelt*m_adelt - wfixsq
            if (temp > m_zero) then
               wsqsav = wfixsq
               step = sqrt(temp/ggfree)
               ggfree = m_zero
               do i = 1, m_nv
                  if (W(i) == bigstp) then
                     temp = m_x_opt(i) - step*m_g_lag(i)
                     if (temp <= m_s_lower(i)) then
                        W(i) = m_s_lower(i) - m_x_opt(i)
                        wfixsq = wfixsq + W(i)**2
                     else if (temp >= m_s_upper(i)) then
                        W(i) = m_s_upper(i) - m_x_opt(i)
                        wfixsq = wfixsq + W(i)**2
                     else
                        ggfree = ggfree + m_g_lag(i)**2
                     end if
                  end if
               end do
               if (wfixsq > wsqsav .and. ggfree > m_zero) then
                  istate = 120
                  cycle
               end if
            end if
            !
            !     Set the remaining free components of W and all components of XALT,
            !     except that W may be scaled later.
            !
            gw = m_zero
            do i = 1, m_nv
               if (W(i) == bigstp) then
                  W(i) = -step*m_g_lag(i)
                  m_x_alt(i) = max(m_s_lower(i), min(m_s_upper(i), m_x_opt(i) + W(i)))
               else if (W(i) == m_zero) then
                  m_x_alt(i) = m_x_opt(i)
               else if (m_g_lag(i) > m_zero) then
                  m_x_alt(i) = m_s_lower(i)
               else
                  m_x_alt(i) = m_s_upper(i)
               end if
               gw = gw + m_g_lag(i)*W(i)
            end do
            !
            !     Set CURV to the curvature of the KNEW-th Lagrange function along W.
            !     Scale W by a factor less than one if that can reduce the modulus of
            !     the Lagrange function at XOPT+W. Set CAUCHY to the final value of
            !     the square of this function.
            !
            curv = dot_product(hcol, matmul(W(1:m_nv), m_xpt)**2)
            if (iflag == 1) curv = -curv
            if (curv > -gw .and. curv < -(m_one + sqrt(2.0_wp))*gw) then
               scale = -gw/curv
               m_x_alt = min(max(m_s_lower, m_x_opt + scale*W(1:m_nv)), m_s_upper)
               m_cauchy = (m_half*gw*scale)**2
            else
               m_cauchy = (gw + m_half*curv)**2
            end if
            !
            !     If IFLAG is zero, then XALT is calculated as before after reversing
            !     the sign of GLAG. Thus two XALT vectors become available. The one that
            !     is chosen is the one that gives the larger value of CAUCHY.
            !
            if (iflag == 0) then
               m_g_lag = -m_g_lag
               x_alt_saved = m_x_alt
               cauchy_saved = m_cauchy
               iflag = 1
               istate = 100
               cycle
            end if
            if (cauchy_saved > m_cauchy) then
               m_x_alt = x_alt_saved
               m_cauchy = cauchy_saved
            end if

            exit

         end select

      end do

   end subroutine altmov

   subroutine prelim(x, calfun)

      implicit none

      real(wp) :: x(1:m_nv), diff, f, fbeg, recip, stepa, stepb
      real(wp) :: rhosq, temp
      integer np, i, j, ii, jj, itemp, nfm, nfx, nf_saved

      procedure(func) :: calfun

      !
      !     The arguments N, NPT, X, XL, XU, RHOBEG, IPRINT and MAXFUN are the
      !       same as the corresponding arguments in SUBROUTINE BOBYQA.
      !     The arguments XBASE, XPT, FVAL, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU
      !       are the same as the corresponding arguments in BOBYQB, the elements
      !       of SL and SU being set in BOBYQA.
      !     GOPT is usually the gradient of the quadratic model at XOPT+XBASE, but
      !       it is set by PRELIM to the gradient of the quadratic model at XBASE.
      !       If XOPT is nonzero, BOBYQB will change it to its usual value later.
      !     NF is maintaned as the number of calls of CALFUN so far.
      !     KOPT will be such that the least calculated value of F so far is at
      !       the point XPT(KOPT,.)+XBASE in the space of the variables.
      !
      !     SUBROUTINE PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
      !     BMAT and ZMAT for the first iteration, and it maintains the values of
      !     NF and KOPT. The vector X is also changed by PRELIM.
      !
      !     Set some constants.
      !
      rhosq = m_rhobeg*m_rhobeg
      recip = m_one/rhosq
      np    = m_nv + 1
      !
      !     Set XBASE to the initial vector of variables, and set the initial
      !     elements of XPT, BMAT, HQ, PQ and ZMAT to zero.
      !
      ! copia x in x_base
      m_x_base = x

      ! azzera matrici e vettori
      m_xpt = 0.0
      m_B   = 0.0
      m_HQ  = 0.0  ! CAMBIATO: matrice piena
      m_pq  = 0.0
      m_Z   = 0.0
      !
      !     Begin the initialization procedure. NF becomes one more than the number
      !     of function values so far. The coordinates of the displacement of the
      !     next initial interpolation point from XBASE are set in XPT(NF+1,.).
      !
      nf_saved = m_nf
      m_nf = 0

      do while (m_nf < m_npt .and. m_nf+nf_saved < m_maxfun)

         nfm = m_nf
         nfx = m_nf - m_nv
         m_nf = m_nf + 1
         if (nfm <= 2*m_nv) then
            if (nfm >= 1 .and. nfm <= m_nv) then
               stepa = m_rhobeg
               if (m_s_upper(nfm) == m_zero) stepa = -stepa
               m_xpt(nfm, m_nf) = stepa
            else if (nfm > m_nv) then
               stepa = m_xpt(nfx, m_nf - m_nv)
               stepb = -m_rhobeg
               if (m_s_lower(nfx) == m_zero) stepb = min(m_two*m_rhobeg, m_s_upper(nfx))
               if (m_s_upper(nfx) == m_zero) stepb = max(-m_two*m_rhobeg, m_s_lower(nfx))
               m_xpt(nfx, m_nf) = stepb
            end if
         else
            itemp = (nfm - np)/m_nv
            jj = nfm - (itemp+1)*m_nv
            ii = jj + itemp
            if (ii > m_nv) then
               itemp = jj
               jj    = ii - m_nv
               ii    = itemp
            end if
            m_xpt(ii,m_nf) = m_xpt(ii,ii+1)
            m_xpt(jj,m_nf) = m_xpt(jj,jj+1)
         end if
         !
         !     Calculate the next value of F. The least function value so far and
         !     its index are required.
         !
         ! clipping base + pt tra lower e upper
         x = m_x_base + m_xpt(:, m_nf)
         call box_project(x)

         ! gestire casi speciali NON SERVE
         !x(1:m_nv) = merge(m_x_lower, x(1:m_nv), m_xpt(:, m_nf) == m_s_lower)
         !x(1:m_nv) = merge(m_x_upper, x(1:m_nv), m_xpt(:, m_nf) == m_s_upper)
         call calfun(m_nv, x, f)
         if (m_iprint == 3) then
            write (*, *) 'Function number', m_nf, '    F =', f
            write (*, *) '    The corresponding X is:', (x(i), i=1, min(m_nv,5))
         end if
         m_f_val(m_nf) = f
         if (m_nf == 1) then
            fbeg = f
            m_kopt = 1
         else if (f < m_f_val(m_kopt)) then
            m_kopt = m_nf
         end if
         !
         !     Set the nonzero initial elements of BMAT and the quadratic model in the
         !     cases when NF is at most 2*N+1. If NF exceeds N+1, then the positions
         !     of the NF-th and (NF-N)-th interpolation points may be switched, in
         !     order that the function value at the first of them contributes to the
         !     off-diagonal second derivative terms of the initial quadratic model.
         !
         if (m_nf <= 2*m_nv + 1) then
            if (m_nf >= 2 .and. m_nf <= m_nv + 1) then
               m_g_opt(nfm) = (f - fbeg)/stepa
               if (m_npt < m_nf + m_nv) then
                  m_B(nfm, 1) = -m_one/stepa
                  m_B(nfm, m_nf) = m_one/stepa
                  m_B(nfm, m_npt + nfm) = -m_half*rhosq
               end if
            else if (m_nf >= m_nv + 2) then
               temp = (f - fbeg)/stepb
               diff = stepb - stepa
               ! CAMBIATO: aggiornamento m_HQ (matrice piena)
               m_HQ(nfx, nfx) = m_two*(temp - m_g_opt(nfx))/diff
               m_g_opt(nfx) = (m_g_opt(nfx)*stepb - temp*stepa)/diff
               if (stepa*stepb < m_zero) then
                  if (f < m_f_val(m_nf - m_nv)) then
                     m_f_val(m_nf) = m_f_val(m_nf - m_nv)
                     m_f_val(m_nf - m_nv) = f
                     if (m_kopt == m_nf) m_kopt = m_nf - m_nv
                     m_xpt(nfx, m_nf - m_nv) = stepb
                     m_xpt(nfx, m_nf) = stepa
                  end if
               end if
               m_B(nfx, 1) = -(stepa + stepb)/(stepa*stepb)
               m_B(nfx, m_nf) = -m_half/m_xpt(nfx, m_nf - m_nv)
               m_B(nfx, m_nf - m_nv) = -m_B(nfx, 1) - m_B(nfx, m_nf)

               m_Z(1, nfx) = sqrt(m_two)/(stepa*stepb)
               m_Z(m_nf, nfx) = sqrt(m_half)/rhosq
               m_Z(m_nf - m_nv, nfx) = -m_Z(1, nfx) - m_Z(m_nf, nfx)
            end if
            !
            !     Set the off-diagonal second derivatives of the Lagrange functions and
            !     the initial quadratic model.
            !
         else
            i = min(ii,jj)  ! CAMBIATO: uso min/max per upper triangular
            j = max(ii,jj)
            m_Z(1, nfx) = recip
            m_Z(m_nf, nfx) = recip
            m_Z(ii+1, nfx) = -recip
            m_Z(jj+1, nfx) = -recip

            write (*, *) 'recip  =', recip

            temp = m_xpt(ii, m_nf)*m_xpt(jj, m_nf)
            ! CAMBIATO: aggiornamento m_HQ (matrice piena upper triangular)
            m_HQ(i, j) = (fbeg - m_f_val(ii+1) - m_f_val(jj+1) + f)/temp
            m_HQ(j, i) = m_HQ(i, j)
         end if

      end do

      m_nf = m_nf + nf_saved 

   end subroutine prelim



    subroutine rescue (n, calfun)

        implicit none

        procedure (func) :: calfun

!
!     The arguments N, NPT, XL, XU, IPRINT, MAXFUN, XBASE, XPT, FVAL, XOPT,
!       GOPT, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU have the same meanings as
!       the corresponding arguments of BOBYQB on the entry to RESCUE.
!     NF is maintained as the number of calls of CALFUN so far, except that
!       NF is set to -1 if the value of MAXFUN prevents further progress.
!     KOPT is maintained so that FVAL(KOPT) is the least calculated function
!       value. Its correct value must be given on entry. It is updated if a
!       new least function value is found, but the corresponding changes to
!       XOPT and GOPT have to be made later by the calling program.
!     DELTA is the current trust region radius.
!     VLAG is a working space vector that will be used for the values of the
!       provisional Lagrange functions at each of the interpolation points.
!       They are part of a product that requires VLAG to be of length NDIM.
!     PTSAUX is also a working space array. For J=1,2,...,N, PTSAUX(1,J) and
!       PTSAUX(2,J) specify the two positions of provisional interpolation
!       points when a nonzero step is taken along e_J (the J-th coordinate
!       direction) through XBASE+XOPT, as specified below. Usually these
!       steps have length DELTA, but other lengths are chosen if necessary
!       in order to satisfy the given bounds on the variables.
!     PTSID is also a working space array. It has NPT components that denote
!       provisional new positions of the original interpolation points, in
!       case changes are needed to restore the linear independence of the
!       interpolation conditions. The K-th point is a candidate for change
!       if and only if PTSID(K) is nonzero. In this case let p and q be the
!       integer parts of PTSID(K) and (PTSID(K)-p) multiplied by N+1. If p
!       and q are both positive, the step from XBASE+XOPT to the new K-th
!       interpolation point is PTSAUX(1,p)*e_p + PTSAUX(1,q)*e_q. Otherwise
!       the step is PTSAUX(1,p)*e_p or PTSAUX(2,q)*e_q in the cases q=0 or
!       p=0, respectively.
!     The first NDIM+NPT elements of the array W are used for working space.
!     The final elements of BMAT and ZMAT are set in a well-conditioned way
!       to the values that are appropriate for the new interpolation points.
!     The elements of GOPT, HQ and PQ are also revised to the values that are
!       appropriate to the final quadratic model.
!
!     Set some constants.
!

        real(wp) :: add, sfrac, bsum, den, diff, distsq, dsqmin, fbase
        real(wp) :: f, hdiag, sum, sumpq, temp, winc, xp, xq, vlmxsq, vquad, m_w1( m_ndim + m_npt )
        real(wp) :: pts_id(m_npt), pts_aux(2,m_nv)
        integer  :: i, j, k, n, jp, jpn, np, ip, iq, iw, kold, nrem, kpt

        write(*,*) 'RESCUE**'

        np = n + 1
        sfrac = m_half / real (np, wp)
        m_nptm = m_npt - np
!
!     Shift the interpolation points so that XOPT becomes the origin, and set
!     the elements of ZMAT to zero. The value of SUMPQ is required in the
!     updating of HQ below. The squares of the distances from XOPT to the
!     other interpolation points are set at the end of W. Increments of WINC
!     may be added later to these squares to balance the consideration of
!     the choice of point that is going to become current.
!
        sumpq = m_zero
        winc = m_zero
        do k = 1, m_npt
            m_xpt(:,k) = m_xpt (:,k) - m_x_opt
            distsq = sum(m_xpt(:,k)** 2)
            sumpq = sumpq + m_pq (k)
            m_w1(m_ndim+k) = distsq
            winc = max (winc, distsq)
        end do
        m_Z = 0
!
!     Update HQ so that HQ and PQ define the second derivatives of the model
!     after XBASE has been shifted to the trust region centre.
!
        m_w1(1:n) = m_half * sumpq * m_x_opt
        do j = 1, n
            m_w1(1:m_npt) = m_w1(1:m_npt) + m_pq * m_xpt(j,:)
            do i = 1, j
                m_HQ(i,j) = m_HQ(i,j) + m_w1(i) * m_x_opt (j) + m_w1(j) * m_x_opt (i)
                m_HQ(j,i) = m_HQ(i,j)
            end do
        end do
!
!     Shift XBASE, SL, SU and XOPT. Set the elements of BMAT to zero, and
!     also set the elements of PTSAUX.
!
        do j = 1, n
            m_x_base (j) = m_x_base (j) + m_x_opt (j)
            m_s_lower (j) = m_s_lower (j) - m_x_opt (j)
            m_s_upper (j) = m_s_upper (j) - m_x_opt (j)
            m_x_opt (j) = m_zero
            pts_aux (1, j) = min (m_delta, m_s_upper(j))
            pts_aux (2, j) = max (-m_delta, m_s_lower(j))
            if (pts_aux(1, j)+pts_aux(2, j) < m_zero) then
                temp = pts_aux (1, j)
                pts_aux (1, j) = pts_aux (2, j)
                pts_aux (2, j) = temp
            end if
            if (abs(pts_aux(2, j)) < m_half*abs(pts_aux(1, j))) then
                pts_aux (2, j) = m_half * pts_aux (1, j)
            end if
        end do
        m_B = 0
        fbase = m_f_val (m_kopt)
!
!     Set the identifiers of the artificial interpolation points that are
!     along a coordinate direction from XOPT, and set the corresponding
!     nonzero elements of BMAT and ZMAT.
!
        pts_id (1) = sfrac
        do j = 1, n
            jp = j + 1
            jpn = jp + n
            pts_id (jp) = real (j, wp) + sfrac
            if (jpn <= m_npt) then
                pts_id (jpn) = real (j, wp) / real (np, wp) + sfrac
                temp = m_one / (pts_aux(1, j)-pts_aux(2, j))
                m_B (j,jp  ) = - temp + m_one / pts_aux (1, j)
                m_B (j,jpn ) = temp + m_one / pts_aux (2, j)
                m_B (j,1   ) = - m_B (j,jp) - m_B (j,jpn)
                m_Z (1, j) = sqrt (2.0_wp) / abs (pts_aux(1, j)*pts_aux(2, j))
                m_Z (jp, j) = m_Z (1, j) * pts_aux (2, j) * temp
                m_Z (jpn, j) = - m_Z (1, j) * pts_aux (1, j) * temp
            else
                m_B (j,1) = - m_one / pts_aux (1, j)
                m_B (j,jp) = m_one / pts_aux (1, j)
                m_B (j,j+m_npt ) = - m_half * pts_aux (1, j) ** 2
            end if
        end do
!
!     Set any remaining identifiers with their nonzero elements of ZMAT.
!
        if (m_npt >= n+np) then
            do k = 2 * np, m_npt
                iw = (real(k-np, wp)-m_half) / real (n, wp)
                ip = k - np - iw * n
                iq = ip + iw
                if (iq > n) iq = iq - n
                pts_id (k) = real (ip, wp) + real (iq, wp) / real (np, wp) + sfrac
                temp = m_one / (pts_aux(1, ip)*pts_aux(1, iq))
                m_Z (1, k-np) = temp
                m_Z (ip+1, k-np) = - temp
                m_Z (iq+1, k-np) = - temp
                m_Z (k, k-np) = temp
            end do
        end if
        nrem   = m_npt
        kold   = 1
        m_knew = m_kopt
!
!     Reorder the provisional points in the way that exchanges PTSID(KOLD)
!     with PTSID(KNEW).
!
80      do j = 1, n
            temp = m_B(j,kold)
            m_B (j,kold) = m_B(j,m_knew)
            m_B (j,m_knew) = temp
        end do
        do j = 1, m_nptm
            temp = m_Z (kold, j)
            m_Z (kold, j) = m_Z (m_knew, j)
            m_Z (m_knew, j) = temp
        end do
        pts_id (kold) = pts_id (m_knew)
        pts_id (m_knew) = m_zero
        m_w1(m_ndim+m_knew) = m_zero
        nrem = nrem - 1
        if (m_knew /= m_kopt) then
            temp = m_v_lag (kold)
            m_v_lag (kold) = m_v_lag (m_knew)
            m_v_lag (m_knew) = temp
!
!     Update the BMAT and ZMAT matrices so that the status of the KNEW-th
!     interpolation point can be changed from provisional to original. The
!     subroutine returns if all the original points are reinstated.
!     The nonnegative values of W(NDIM+K) are required in the search below.
!
            call update ()
            if (nrem == 0) return
            do k = 1, m_npt
                m_w1(m_ndim+k) = abs (m_w1(m_ndim+k))
            end do
        end if
!
!     Pick the index KNEW of an original interpolation point that has not
!     yet replaced one of the provisional interpolation points, giving
!     attention to the closeness to XOPT and to previous tries with KNEW.
!
120     dsqmin = m_zero
        do k = 1, m_npt
            if (m_w1(m_ndim+k) > m_zero) then
                if (dsqmin == m_zero .or. m_w1(m_ndim+k) < dsqmin) then
                    m_knew = k
                    dsqmin = m_w1(m_ndim+k)
                end if
            end if
        end do
        if (dsqmin == m_zero) go to 260
!
!     Form the W-vector of the chosen original interpolation point.
!
        m_w1(m_npt+1:m_npt+n) = m_xpt(:,m_knew)
        do k = 1, m_npt
            add = 0
            if (k == m_kopt) then
                continue
            else if (pts_id(k) == m_zero) then
               add = add + sum( m_w1(m_npt+1:m_npt+n) * m_xpt(:,k) )
            else
                ip = pts_id (k)
                if (ip > 0) add = m_w1(m_npt+ip) * pts_aux (1, ip)
                iq = real (np, wp) * pts_id (k) - real (ip*np, wp)
                if (iq > 0) then
                    iw = 1
                    if (ip == 0) iw = 2
                    add = add + m_w1(m_npt+iq) * pts_aux (iw, iq)
                end if
            end if
            m_w1(k) = m_half * add * add
        end do
!
!     Calculate VLAG and BETA for the required updating of the H matrix if
!     XPT(KNEW,.) is reinstated in the set of interpolation points.
!
        do k = 1, m_npt
            m_v_lag (k) = sum( m_B(:,k) * m_w1(m_npt+1:m_npt+n) )
        end do
        m_beta = m_zero
        do j = 1, m_nptm
            add = sum( m_Z(:,j) * m_w1(1:m_npt) )
            m_beta = m_beta - add * add
            m_v_lag (1:m_npt) = m_v_lag (1:m_npt) + add * m_Z(:, j)
        end do
        bsum = m_zero
        do j = 1, n
            add = sum( m_B(j,1:m_npt) * m_w1(1:m_npt) )
            jp = j + m_npt
            bsum = bsum + add * m_w1(jp)
            add = add + sum( m_B(j,m_npt+1:m_ndim) * m_w1(m_npt+1:m_ndim) )
            bsum = bsum + add * m_w1(jp)
            m_v_lag (jp) = add
        end do
        distsq = sum( m_xpt(:,m_knew) ** 2 )
        m_beta = m_half * distsq * distsq + m_beta - bsum
        m_v_lag (m_kopt) = m_v_lag (m_kopt) + m_one
!
!     KOLD is set to the index of the provisional interpolation point that is
!     going to be deleted to make way for the KNEW-th original interpolation
!     point. The choice of KOLD is governed by the avoidance of a small value
!     of the denominator in the updating calculation of UPDATE.
!
        m_denom = m_zero
        vlmxsq = m_zero
        do k = 1, m_npt
            if (pts_id(k) /= m_zero) then
                hdiag = m_zero
                do j = 1, m_nptm
                    hdiag = hdiag + m_Z (k, j) ** 2
                end do
                den = m_beta * hdiag + m_v_lag (k) ** 2
                if (den > m_denom) then
                    kold = k
                    m_denom = den
                end if
            end if
            vlmxsq = max (vlmxsq, m_v_lag(k)**2)
        end do
        if (m_denom <= 1.0e-2_wp*vlmxsq) then
            m_w1(m_ndim+m_knew) = - m_w1(m_ndim+m_knew) - winc
            go to 120
        end if
        go to 80
!
!     When label 260 is reached, all the final positions of the interpolation
!     points have been chosen although any changes have not been included yet
!     in XPT. Also the final BMAT and ZMAT matrices are complete, but, apart
!     from the shift of XBASE, the updating of the quadratic model remains to
!     be done. The following cycle through the new interpolation points begins
!     by putting the new point in XPT(KPT,.) and by setting PQ(KPT) to zero,
!     except that a RETURN occurs if MAXFUN prohibits another value of F.
!
260     do kpt = 1, m_npt
            if (pts_id(kpt) == m_zero) cycle
            if (m_nf >= m_maxfun) then
                m_nf = - 1
                return
            end if
            do j = 1, n
                m_w1(j) = m_xpt (j,kpt)
                m_xpt (j,kpt) = m_zero
                temp = m_pq (kpt) * m_w1(j)
                do i = 1, j
                    m_HQ(i,j) = m_HQ(i,j) + temp * m_w1(i)
                    m_HQ(j,i) = m_HQ(i,j)
                end do
            end do
            m_pq (kpt) = m_zero
            ip = pts_id (kpt)
            iq = real (np, wp) * pts_id (kpt) - real (ip*np, wp)
            if (ip > 0) then
                xp = pts_aux (1,ip)
                m_xpt (ip,kpt) = xp
            end if
            if (iq > 0) then
                xq = pts_aux (1, iq)
                if (ip == 0) xq = pts_aux (2, iq)
                m_xpt (iq,kpt) = xq
            end if
!
!     Set VQUAD to the value of the current model at the new point.
!
            vquad = fbase
            if (ip > 0) then
                vquad = vquad + xp * (m_g_opt(ip)+m_half*xp*m_HQ(ip,ip))
            end if
            if (iq > 0) then
                vquad = vquad + xq * (m_g_opt(iq)+m_half*xq*m_HQ(iq,iq))
                if (ip > 0) then
                    vquad = vquad + xp * xq * m_HQ(ip,iq)
                end if
            end if
            do k = 1, m_npt
                temp = m_zero
                if (ip > 0) temp = temp + xp * m_xpt (ip, k)
                if (iq > 0) temp = temp + xq * m_xpt (iq, k)
                vquad = vquad + m_half * m_pq (k) * temp * temp
            end do
!
!     Calculate F at the new interpolation point, and set DIFF to the factor
!     that is going to multiply the KPT-th Lagrange function when the model
!     is updated to provide interpolation to the new function value.
!
            m_w1(1:n) = min(max(m_x_base+m_xpt(:,kpt),m_x_lower),m_x_upper)
            do i = 1, n
                if (m_xpt(i,kpt) == m_s_lower(i)) m_w1(i) = m_x_lower(i)
                if (m_xpt(i,kpt) == m_s_upper(i)) m_w1(i) = m_x_upper (i)
            end do
            m_nf = m_nf + 1
            call calfun (n, m_w1(1:n), f)
            if (m_iprint == 3) then
                write(*,*) 'Function number N.', m_nf, '    F =', f
                write(*,*) '    The corresponding X is:', m_w1(1:min(m_nv,5))
            end if
            m_f_val (kpt) = f
            if (f < m_f_val(m_kopt)) m_kopt = kpt
            diff = f - vquad
!
!     Update the quadratic model. The RETURN from the subroutine occurs when
!     all the new interpolation points are included in the model.
!
            m_g_opt = m_g_opt + diff * m_B(:,kpt)
            do k = 1, m_npt
                temp = diff * sum( m_Z(k,:) * m_Z(kpt,:))
                if (pts_id(k) == m_zero) then
                    m_pq (k) = m_pq (k) + temp
                else
                    ip = pts_id (k)
                    iq = real (np, wp) * pts_id (k) - real (ip*np, wp)
                    if (ip == 0) then
                        m_HQ(iq,iq) = m_HQ(iq,iq) + temp * pts_aux (2, iq) ** 2
                    else
                        m_HQ(ip,ip) = m_HQ(ip,ip) + temp * pts_aux (1, ip) ** 2
                        if (iq > 0) then
                            m_HQ(iq,iq) = m_HQ (iq,iq) + temp * pts_aux (1, iq) ** 2
                            m_HQ(ip,iq) = m_HQ(ip,iq) + temp * pts_aux (1, ip) * pts_aux (1, iq)
                            m_HQ(iq,ip) = m_HQ(ip,iq)
                        end if
                    end if
                end if
            end do
            pts_id (kpt) = m_zero
        end do

    end subroutine rescue



   subroutine trsbox()
      implicit none

      ! Variabili locali
      integer :: i, iact, isav, itermax, iu, itcsav, iterc, nact
      real(wp) :: onemin, angbd, blen, cth, angt, delsq, dhd, dhs, dredg, dredsq, ds
      real(wp) :: ggsav, gredsq, tempa, tempb, qred, ratio, rdnext, rdprev, redmax, rednew, redsav, resid
      real(wp) :: ssq, sdec, shs, sqstp, sth, stplen, temp, xsav, sredg, stepsq

      ! Stati della macchina a stati
      integer, parameter :: S_INIT = 0, S_TCG_INIT = 1, S_TCG_LOOP = 2, S_TCG_STEPLENGTH = 3
      integer, parameter :: S_TCG_UPDATE = 4, S_BOUNDARY_INIT = 5, S_BOUNDARY_LOOP = 6
      integer, parameter :: S_ALTERNATING_LOOP = 7, S_FINAL = 8
      integer :: state

      ! Vettori temporanei
      real(wp) :: d(m_nv)
      logical :: mask_lower(m_nv), mask_upper(m_nv)

      ! --- Dichiarazione Alias (Puntatori) ---
      ! Rimuovi 'allocatable, target' e usa solo 'pointer'
      real(wp), pointer :: v_lag0(:), v_lag1(:)
      real(wp), pointer :: B0(:, :), B1(:, :)

      ! --- Associazione Alias ---
      v_lag0 => m_v_lag(1:m_npt)
      v_lag1 => m_v_lag(m_npt + 1:m_npt + m_nv)
      B0 => m_B(1:m_nv, 1:m_npt)
      B1 => m_B(1:m_nv, m_npt + 1:m_npt + m_nv)

      ! =====================================================================
      ! 1. INIZIALIZZAZIONE: PROIEZIONE GRADIENTE E VARIABILI ATTIVE
      ! =====================================================================
      ! g_proj = P_{[L,U]}(x_opt - g_opt) - x_opt (ma implementato come controllo segno)
      onemin = -1.0_wp
      iterc = 0
      sqstp = 0.0_wp
      d = 0.0_wp
      m_g_new = m_g_opt

      ! Identificazione variabili attive (al limite con gradiente che punta fuori)
      ! x_i al lower bound e g_i ≥ 0 → variabile attiva
      ! x_i al upper bound e g_i ≤ 0 → variabile attiva
      mask_lower = (m_x_opt <= m_s_lower + epsilon(1.0_wp)) .and. (m_g_opt >= 0.0_wp)
      mask_upper = (m_x_opt >= m_s_upper - epsilon(1.0_wp)) .and. (m_g_opt <= 0.0_wp)

      ! XBDI: -1 = attiva al lower bound, +1 = attiva al upper bound, 0 = libera
      m_xbdi = 0
      where (mask_lower) m_xbdi = -1
      where (mask_upper) m_xbdi = 1

      nact = count(m_xbdi /= 0)
      delsq = m_delta**2
      qred = 0.0_wp
      m_crvmin = onemin

      ! Inizia la fase TCG (Trust Region Conjugate Gradient)
      state = S_TCG_INIT

      ! =====================================================================
      ! 2. MACCHINA A STATI PRINCIPALE
      ! =====================================================================
      do

         select case (state)

            ! =================================================================
         case (S_TCG_INIT)  ! Inizializzazione/restart TCG
            ! =================================================================
            ! β ← 0 (reset parametro CG)
            m_beta = 0.0_wp
            state = S_TCG_LOOP

            ! =================================================================
         case (S_TCG_LOOP)  ! Iterazione TCG principale
            ! =================================================================
            ! -------------------------------------------------------------
            ! 2.1 CALCOLO DIREZIONE CONIUGATA: s ← β·s - g_new
            ! -------------------------------------------------------------
            ! s_{k+1} = β_k·s_k - ∇Q(x_k)
            m_s = m_beta*m_s - m_g_new
            where (m_xbdi /= 0) m_s = 0.0_wp  ! Annulla componenti variabili attive

            stepsq = dot_product(m_s, m_s)

            ! Criteri di arresto TCG
            if (stepsq == 0.0_wp) then
               state = S_FINAL
               cycle
            end if

            if (m_beta == 0.0_wp) then
               gredsq = stepsq
               itermax = iterc + m_nv - nact  ! Massimo numero iterazioni
            end if

            ! Criterio di Cauchy: gredsq·Δ² ≤ 10⁻⁴·qred²
            if (gredsq*delsq <= 1.0e-4_wp*qred**2) then
               state = S_FINAL
               cycle
            end if

            ! -------------------------------------------------------------
            ! 2.2 PRODOTTO HESSIANO: Hs ← H·s
            ! -------------------------------------------------------------
            ! H = HQ (triang. sup.) + Σ_{k=1}^{npt} pq_k·x_k·x_k^T
            m_hs = grad_times_vector(m_s)

            state = S_TCG_STEPLENGTH

            ! =================================================================
         case (S_TCG_STEPLENGTH)  ! Calcolo lunghezza passo
            ! =================================================================
            ! -------------------------------------------------------------
            ! 2.3 CALCOLO LUNGHEZZA PASSO α (Steihaug-Toint)
            ! -------------------------------------------------------------
            ! α = argmin_{0≤τ≤τ_max} Q(x + τ·s) s.t. ||d+τ·s|| ≤ Δ
            ! con τ_max = min(α_Cauchy, α_bounds, α_neg_curv)
            ! -------------------------------------------------------------

            ! Residuo norma: Δ² - ||d||²
            ! Calcola usando maschere logiche
            resid = delsq - sum(d**2, mask=(m_xbdi == 0))
            ds = sum(m_s*d, mask=(m_xbdi == 0))
            shs = sum(m_s*m_hs, mask=(m_xbdi == 0))

            ! Caso: sfera già saturata (||d|| ≥ Δ)
            if (resid <= 0.0_wp) then
               state = S_BOUNDARY_INIT
               cycle
            end if

            ! -------------------------------------------------------------
            ! 2.4 CALCOLO PASSO MASSIMO SULLA SFERA (α_sphere)
            ! -------------------------------------------------------------
            ! Soluzione di: ||d + α·s|| = Δ
            temp = sqrt(stepsq*resid + ds**2)
            if (ds < 0.0_wp) then
               blen = (temp - ds)/stepsq
            else
               blen = resid/(temp + ds)
            end if
            stplen = blen

            ! -------------------------------------------------------------
            ! 2.5 CONTROLLO CURVATURA: se s^T·H·s > 0 → minimo lungo s
            ! -------------------------------------------------------------
            if (shs > 0.0_wp) then
               stplen = min(blen, gredsq/shs)  ! α_Cauchy
            end if

            ! -------------------------------------------------------------
            ! 2.6 CONTROLLO VINCOLI DI SCATOLA (α_bounds)
            ! -------------------------------------------------------------
            call box_step(m_s, d, iact, temp)
            if (temp < stplen) then
               stplen = temp
            else
               iact = 0
            end if

            state = S_TCG_UPDATE

            ! =================================================================
         case (S_TCG_UPDATE)  ! Aggiornamento dopo passo TCG
            ! =================================================================
            ! -------------------------------------------------------------
            ! 2.7 AGGIORNAMENTO VARIABILI DOPO PASSO α
            ! -------------------------------------------------------------
            sdec = 0.0_wp
            if (stplen > 0.0_wp) then
               iterc = iterc + 1

               ! Stima curvatura minima
               temp = shs/stepsq
               if (iact == 0 .and. temp > 0.0_wp) then
                  m_crvmin = min(m_crvmin, temp)
                  if (m_crvmin == onemin) m_crvmin = temp
               end if

               ! Salva norma gradiente prima dell'aggiornamento
               ggsav = gredsq
               gredsq = 0.0_wp

               ! Aggiornamento: x ← x + α·s, g ← g + α·H·s
               !$omp parallel do reduction(+:gredsq)
               do i = 1, m_nv
                  m_g_new(i) = m_g_new(i) + stplen*m_hs(i)
                  if (m_xbdi(i) == 0) gredsq = gredsq + m_g_new(i)**2
                  d(i) = d(i) + stplen*m_s(i)
               end do
               !$omp end parallel do

               ! Riduzione modello quadratico: ΔQ = α·(g^T·s) - ½·α²·(s^T·H·s)
               sdec = max(stplen*(ggsav - 0.5_wp*stplen*shs), 0.0_wp)
               qred = qred + sdec
            end if

            ! -------------------------------------------------------------
            ! 2.8 DECISIONE TRANSIZIONE STATO
            ! -------------------------------------------------------------
            if (iact > 0) then
               ! TCG ha incontrato vincolo → fissa variabile e restart
               nact = nact + 1
               m_xbdi(iact) = sign(1.0_wp, -m_s(iact))
               delsq = delsq - d(iact)**2

               if (delsq <= 0.0_wp) then
                  state = S_BOUNDARY_INIT
               else
                  state = S_TCG_INIT
               end if

            else if (stplen < blen) then
               ! TCG continua (non ha toccato né sfera né vincoli)
               if (iterc == itermax .or. sdec <= 0.01_wp*qred) then
                  state = S_FINAL
               else
                  ! Fletcher-Reeves: β = ||g_{k+1}||² / ||g_k||²
                  m_beta = gredsq/ggsav
                  state = S_TCG_LOOP
               end if
            else
               ! TCG ha saturato la sfera → passa a Boundary Phase
               state = S_BOUNDARY_INIT
            end if

            ! =================================================================
         case (S_BOUNDARY_INIT)  ! Inizializzazione fase Boundary
            ! =================================================================
            m_crvmin = 0.0_wp

            if (nact >= m_nv - 1) then
               state = S_FINAL
               cycle
            end if

            state = S_BOUNDARY_LOOP

            ! =================================================================
         case (S_BOUNDARY_LOOP)  ! Loop principale Boundary Phase
            ! =================================================================
            if (nact >= m_nv - 1) then
               state = S_FINAL
               cycle
            end if

            ! -------------------------------------------------------------
            ! 3.1 CALCOLO GRADIENTE RIDOTTO E DIREZIONE TANGENTE
            ! -------------------------------------------------------------
            ! d: passo accumulato, s: direzione tangente al vincolo
            dredsq = 0.0_wp
            dredg = 0.0_wp
            gredsq = 0.0_wp

            !$omp parallel do reduction(+:dredsq, dredg, gredsq)
            do i = 1, m_nv
               if (m_xbdi(i) == 0) then
                  dredsq = dredsq + d(i)**2
                  dredg = dredg + d(i)*m_g_new(i)
                  gredsq = gredsq + m_g_new(i)**2
                  m_s(i) = d(i)  ! Inizializza s come d
               else
                  m_s(i) = 0.0_wp
               end if
            end do
            !$omp end parallel do

            itcsav = iterc

            ! -------------------------------------------------------------
            ! 3.2 CALCOLO H·s E INIZIALIZZAZIONE HRED
            ! -------------------------------------------------------------
            m_hs = grad_times_vector(m_s)

            if (iterc == itcsav) then
               m_hred = m_hs  ! HRED = H·d (per uso successivo)
            end if

            state = S_ALTERNATING_LOOP

            ! =================================================================
         case (S_ALTERNATING_LOOP)  ! Loop alternativo (CG sul bordo)
            ! =================================================================
            iterc = iterc + 1

            ! -------------------------------------------------------------
            ! 3.3 CALCOLO NUOVA DIREZIONE CONIUGATA SUL BORDO
            ! -------------------------------------------------------------
            ! s = ( (d^T·g)·d - ||d||²·g ) / sqrt( ||d||²·||g||² - (d^T·g)² )
            ! (direzione di massima riduzione lungo il bordo)
            temp = gredsq*dredsq - dredg**2

            if (temp <= 1.0e-4_wp*qred**2) then
               state = S_FINAL
               cycle
            end if

            temp = sqrt(temp)
            !$omp parallel do
            do i = 1, m_nv
               if (m_xbdi(i) == 0) then
                  m_s(i) = (dredg*d(i) - dredsq*m_g_new(i))/temp
               else
                  m_s(i) = 0.0_wp
               end if
            end do
            !$omp end parallel do

            sredg = -temp  ! s^T·g = -||s||·||g||·sin(θ)

            ! -------------------------------------------------------------
            ! 3.4 CALCOLO ANGOLO MASSIMO (ANGBD) SENZA USCIRE DALLA SCATOLA
            ! -------------------------------------------------------------
            angbd = 1.0_wp
            iact = 0

            do i = 1, m_nv
               if (m_xbdi(i) == 0) then
                  tempa = m_x_opt(i) + d(i) - m_s_lower(i)  ! Distanza da lower bound
                  tempb = m_s_upper(i) - m_x_opt(i) - d(i)  ! Distanza da upper bound

                  ! Controlla se già al limite
                  if (tempa <= 0.0_wp) then
                     nact = nact + 1
                     m_xbdi(i) = -1.0_wp
                     state = S_BOUNDARY_LOOP
                     exit
                  else if (tempb <= 0.0_wp) then
                     nact = nact + 1
                     m_xbdi(i) = 1.0_wp
                     state = S_BOUNDARY_LOOP
                     exit
                  end if

                  ratio = 1.0_wp
                  ssq = d(i)**2 + m_s(i)**2  ! ||d+θ·s||²

                  ! Controllo lower bound
                  temp = ssq - (m_x_opt(i) - m_s_lower(i))**2
                  if (temp > 0.0_wp) then
                     temp = sqrt(temp) - m_s(i)
                     if (angbd*temp > tempa) then
                        angbd = tempa/temp
                        iact = i
                        xsav = -1.0_wp
                     end if
                  end if

                  ! Controllo upper bound
                  temp = ssq - (m_s_upper(i) - m_x_opt(i))**2
                  if (temp > 0.0_wp) then
                     temp = sqrt(temp) + m_s(i)
                     if (angbd*temp > tempb) then
                        angbd = tempb/temp
                        iact = i
                        xsav = 1.0_wp
                     end if
                  end if
               end if
            end do

            if (state == S_BOUNDARY_LOOP) cycle

            ! -------------------------------------------------------------
            ! 3.5 CALCOLO H·s E PRODOTTI SCALARI
            ! -------------------------------------------------------------
            m_hs = grad_times_vector(m_s)

            shs = 0.0_wp
            dhs = 0.0_wp
            dhd = 0.0_wp

            !$omp parallel do reduction(+:shs, dhs, dhd)
            do i = 1, m_nv
               if (m_xbdi(i) == 0) then
                  shs = shs + m_s(i)*m_hs(i)
                  dhs = dhs + d(i)*m_hs(i)
                  dhd = dhd + d(i)*m_hred(i)
               end if
            end do
            !$omp end parallel do

            ! -------------------------------------------------------------
            ! 3.6 RICERCA LINEARE SUL CERCHIO: max_{0≤θ≤θ_max} Q(d·cosθ + s·sinθ)
            ! -------------------------------------------------------------
            ! Discretizzazione angolo: θ_i = θ_max·i/17, i=1..17
            redmax = 0.0_wp
            isav = 0
            redsav = 0.0_wp
            iu = int(17.0_wp*angbd + 3.1_wp)

            do i = 1, iu
               angt = angbd*real(i, wp)/real(iu, wp)
               sth = (2.0_wp*angt)/(1.0_wp + angt**2)  ! sin(2·arctan(θ))
               temp = shs + angt*(angt*dhd - 2.0_wp*dhs)

               ! ΔQ = sinθ·(θ·d^T·g - s^T·g - ½·sinθ·(s^T·H·s + θ²·d^T·H·d - 2θ·d^T·H·s))
               rednew = sth*(angt*dredg - sredg - 0.5_wp*sth*temp)

               if (rednew > redmax) then
                  redmax = rednew
                  isav = i
                  rdprev = redsav
               else if (i == isav + 1) then
                  rdnext = rednew
               end if
               redsav = rednew
            end do

            ! -------------------------------------------------------------
            ! 3.7 AGGIORNAMENTO PASSO CON INTERPOLAZIONE PARABOLICA
            ! -------------------------------------------------------------
            if (isav == 0) then
               state = S_FINAL
               cycle
            end if

            ! Interpolazione parabolica per raffinare θ_opt
            if (isav < iu) then
               temp = (rdnext - rdprev)/(2.0_wp*redmax - rdprev - rdnext)
               angt = angbd*(real(isav, wp) + 0.5_wp*temp)/real(iu, wp)
            end if

            ! Coordinate polari: d_new = d·cosφ + s·sinφ, con φ = 2·arctan(θ)
            cth = (1.0_wp - angt**2)/(1.0_wp + angt**2)  ! cos(φ)
            sth = (2.0_wp*angt)/(1.0_wp + angt**2)      ! sin(φ)

            temp = shs + angt*(angt*dhd - 2.0_wp*dhs)
            sdec = sth*(angt*dredg - sredg - 0.5_wp*sth*temp)

            if (sdec <= 0.0_wp) then
               state = S_FINAL
               cycle
            end if

            ! -------------------------------------------------------------
            ! 3.8 AGGIORNAMENTO VARIABILI
            ! -------------------------------------------------------------
            ! g_new ← g_new + (cosφ-1)·H·d + sinφ·H·s
            ! hred ← cosφ·H·d + sinφ·H·s
            m_g_new = m_g_new + (cth - 1.0_wp)*m_hred + sth*m_hs
            m_hred = cth*m_hred + sth*m_hs

            dredg = 0.0_wp
            gredsq = 0.0_wp

            !$omp parallel do reduction(+:dredg, gredsq)
            do i = 1, m_nv
               if (m_xbdi(i) == 0) then
                  d(i) = cth*d(i) + sth*m_s(i)
                  dredg = dredg + d(i)*m_g_new(i)
                  gredsq = gredsq + m_g_new(i)**2
               end if
            end do
            !$omp end parallel do

            qred = qred + sdec

            ! -------------------------------------------------------------
            ! 3.9 DECISIONE TRANSIZIONE
            ! -------------------------------------------------------------
            if (iact > 0 .and. isav == iu) then
               ! Ha toccato un nuovo vincolo → fissa variabile
               nact = nact + 1
               m_xbdi(iact) = xsav
               state = S_BOUNDARY_LOOP
            else if (sdec > 0.01_wp*qred) then
               ! Continua alternation loop
               state = S_ALTERNATING_LOOP
            else
               state = S_FINAL
            end if

         case (S_FINAL)
            exit
         end select

      end do

      ! =====================================================================
      ! 4. FINALIZZAZIONE: CALCOLO NUOVO PUNTO E CONTROLLO VINCOLI
      ! =====================================================================
      ! x_new = P_{[L,U]}(x_opt + d)  (proiezione su scatola)
      m_x_new = m_x_opt + d
      m_x_new = max(min(m_x_new, m_s_upper), m_s_lower)

      ! Calcola passo effettivo (dopo proiezione)
      d = m_x_new - m_x_opt
      m_dsq = dot_product(d, d)

   end subroutine trsbox

   subroutine update()

      implicit none

      integer j, jp
      real(wp) :: t, c, s, tau, tempa, tempb, ztest

      real(wp) :: W(m_npt + m_nv)

      !
      !     The arrays BMAT and ZMAT are updated, as required by the new position
      !     of the interpolation point that has the index KNEW. The vector VLAG has
      !     N+NPT components, set on entry to the first NPT and last N components
      !     of the product Hw in equation (4.11) of the Powell (2006) paper on
      !     NEWUOA. Further, BETA is set on entry to the value of the parameter
      !     with that name, and DENOM is set to the denominator of the updating
      !     formula. Elements of ZMAT may be treated as zero if their moduli are
      !     at most ZTEST. The first NDIM elements of W are used for working space.
      !
      !     Set some constants.
      !
      ztest = 1.0e-20_wp*maxval(abs(m_Z))
      !
      !     Apply the rotations that put zeros in the KNEW-th row of ZMAT.
      !
      do j = 2, m_nptm
         if (abs(m_Z(m_knew, j)) > ztest) then
            c = m_Z(m_knew, 1)
            s = m_Z(m_knew, j)
            t = hypot(c, s)
            c = c/t
            s = s/t
            W(1:m_npt) = c*m_Z(:, 1) + s*m_Z(:, j)
            m_Z(:, j) = c*m_Z(:, j) - s*m_Z(:, 1)
            m_Z(:, 1) = W(1:m_npt)
         end if
         m_Z(m_knew, j) = 0.0
      end do
      !
      !     Put the first NPT components of the KNEW-th column of HLAG into W,
      !     and calculate the parameters of the updating formula.
      !
      W(1:m_npt) = m_Z(m_knew, 1)*m_Z(:, 1)
      m_alpha = W(m_knew)
      tau = m_v_lag(m_knew)
      m_v_lag(m_knew) = m_v_lag(m_knew) - m_one
      !
      !     Complete the updating of ZMAT.
      !
      t = sqrt(m_denom)
      tempb = m_Z(m_knew, 1)/t
      tempa = tau/t
      m_Z(:, 1) = tempa*m_Z(:, 1) - tempb*m_v_lag(1:m_npt)
      !
      !     Finally, update the matrix BMAT.
      !
      do j = 1, m_nv
         jp = m_npt + j

         W(jp) = m_B(j, m_knew); 
         tempa = (m_alpha*m_v_lag(jp) - tau*W(jp))/m_denom
         tempb = (-m_beta*W(jp) - tau*m_v_lag(jp))/m_denom

         m_B(j, 1:jp) = m_B(j, 1:jp) + tempa*m_v_lag(1:jp) + tempb*W(1:jp)
         m_B(1:j, jp) = m_B(j, m_npt + 1:jp)
      end do

   end subroutine update

   !*****************************************************************************************
   !>
   !  Test problem for [[bobyqa]] with the Rosenbrock function.
   !  Starting point: x = [-4, 4]
   !  Bounds: xl = [-5, -5], xu = [5, 5]
   !  Detailed printing (m_iprint=3) for maximum information.
   !  The algorithm will print information at each function evaluation and when RHO is reduced.

   subroutine test_rosenbrock()

      implicit none

      integer, parameter :: n = 2
      real(wp), dimension(n) :: x, xl, xu
      integer :: npt, iprint, maxfun
      real(wp) :: rhobeg, rhoend
      integer :: i

      ! Set the parameters
      npt = 2*n + 1  ! Recommended value
      iprint = 3     ! Maximum output
      maxfun = 500   ! Maximum number of function evaluations
      rhobeg = 0.1_wp   ! Initial trust region radius
      rhoend = 1.0e-6_wp ! Final trust region radius

      ! Set bounds
      xl = -4.1_wp
      xu = 1_wp

      ! Initial point
      x = [-4.0_wp, 4.0_wp]

      write (*, *) '==============================================='
      write (*, *) 'BOBYQA Test: Rosenbrock Function Optimization'
      write (*, *) '==============================================='
      write (*, *)
      write (*, *) 'Problem dimension (n): ', n
      write (*, *) 'Number of interpolation points (npt): ', npt
      write (*, *) 'Initial trust region radius (rhobeg): ', rhobeg
      write (*, *) 'Final trust region radius (rhoend): ', rhoend
      write (*, *) 'Maximum function evaluations (maxfun): ', maxfun
      write (*, *) 'Printing level (iprint): ', iprint
      write (*, *)
      write (*, *) 'Initial point:'
      do i = 1, n
         write (*, *) '  x(', i, ') = ', x(i)
      end do
      write (*, *)
      write (*, *) 'Lower bounds:'
      do i = 1, n
         write (*, *) '  xl(', i, ') = ', xl(i)
      end do
      write (*, *)
      write (*, *) 'Upper bounds:'
      do i = 1, n
         write (*, *) '  xu(', i, ') = ', xu(i)
      end do
      write (*, *)
      write (*, *) 'Initial function value: ', rosenbrock(x)
      write (*, *) '==============================================='
      write (*, *)

      ! Call BOBYQA optimizer
      call bobyqa(n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun, calfun_rosenbrock)

      write (*, *)
      write (*, *) '==============================================='
      write (*, *) 'Optimization Results:'
      write (*, *) '==============================================='
      write (*, *) 'Optimal point found:'
      do i = 1, n
         write (*, *) '  x(', i, ') = ', x(i)
      end do
      write (*, *)
      write (*, *) 'Function value at optimal point: ', rosenbrock(x)
      write (*, *) '==============================================='

   contains

      ! Rosenbrock function wrapper for BOBYQA
      subroutine calfun_rosenbrock(n, x, f)
         integer, intent(in) :: n
         real(wp), dimension(:), intent(in) :: x
         real(wp), intent(out) :: f
         f = rosenbrock(x)
      end subroutine calfun_rosenbrock

     function rosenbrock(x) result(f)
        real(wp), intent(in) :: x(:)
        real(wp) :: f
        f = 100.0_wp*(x(2) - x(1)**2)**2 + (1.0_wp - x(1))**2
     end function rosenbrock

   end subroutine test_rosenbrock
   !*****************************************************************************************

   !*****************************************************************************************
!*****************************************************************************************
!>
!  Test problem for [[bobyqa]] with the Freudenstein-Roth 2D function.
!  Starting point: x = [0.5, -2.0]
!  Bounds: xl = [-10, -10], xu = [10, 10]
!  Detailed printing (iprint=3) for maximum information.
!  Global minimum at x = [5.0, 4.0]

   subroutine test_freudenstein_roth()

      implicit none

      integer, parameter :: n = 2
      real(wp), dimension(n) :: x, xl, xu
      integer :: npt, iprint, maxfun
      real(wp) :: rhobeg, rhoend
      integer :: i

      ! Set the parameters
      npt = 2*n + 1
      iprint = 3
      maxfun = 500
      rhobeg = 0.1_wp
      rhoend = 1.0e-6_wp

      ! Set bounds
      xl = -10.0_wp
      xu = 10.0_wp

      ! Initial point
      !x = [ 0.5_wp, -2.0_wp ]
      x = [0.5_wp, 2.0_wp]

      write (*, *) '======================================================'
      write (*, *) 'BOBYQA Test: Freudenstein-Roth 2D Function Optimization'
      write (*, *) '======================================================'
      write (*, *)
      write (*, *) 'Problem dimension (n): ', n
      write (*, *) 'Number of interpolation points (npt): ', npt
      write (*, *) 'Initial trust region radius (rhobeg): ', rhobeg
      write (*, *) 'Final trust region radius (rhoend): ', rhoend
      write (*, *) 'Maximum function evaluations (maxfun): ', maxfun
      write (*, *) 'Printing level (iprint): ', iprint
      write (*, *)

      write (*, *) 'Initial point:'
      do i = 1, n
         write (*, *) '  x(', i, ') = ', x(i)
      end do
      write (*, *)

      write (*, *) 'Lower bounds:'
      do i = 1, n
         write (*, *) '  xl(', i, ') = ', xl(i)
      end do
      write (*, *)

      write (*, *) 'Upper bounds:'
      do i = 1, n
         write (*, *) '  xu(', i, ') = ', xu(i)
      end do
      write (*, *)

      write (*, *) 'Initial function value: ', freudenstein_roth(x)
      write (*, *) '======================================================'
      write (*, *)

      ! Call BOBYQA optimizer
      call bobyqa(n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun, calfun_freudenstein_roth)

      write (*, *)
      write (*, *) '======================================================'
      write (*, *) 'Optimization Results:'
      write (*, *) '======================================================'
      write (*, *) 'Optimal point found:'
      do i = 1, n
         write (*, *) '  x(', i, ') = ', x(i)
      end do
      write (*, *)

      write (*, *) 'Function value at optimal point: ', freudenstein_roth(x)
      write (*, *) '======================================================'

   contains

      !--------------------------------------------------------------------
      ! Freudenstein-Roth objective function
      !--------------------------------------------------------------------
      real(wp) function freudenstein_roth(x)
         real(wp), intent(in) :: x(:)
         real(wp) :: x1, x2, f1, f2

         x1 = x(1)
         x2 = x(2)

         f1 = -13.0_wp + x1 + ((5.0_wp - x2)*x2 - 2.0_wp)*x2
         f2 = -29.0_wp + x1 + ((x2 + 1.0_wp)*x2 - 14.0_wp)*x2

         freudenstein_roth = f1*f1 + f2*f2
      end function freudenstein_roth

      !--------------------------------------------------------------------
      ! Wrapper for BOBYQA
      !--------------------------------------------------------------------
      subroutine calfun_freudenstein_roth(n, x, f)
         integer, intent(in) :: n
         real(wp), intent(in) :: x(:)
         real(wp), intent(out) :: f

         f = freudenstein_roth(x)
      end subroutine calfun_freudenstein_roth

   end subroutine test_freudenstein_roth

!*****************************************************************************************
!>
!  Test "cattivo" per [[bobyqa]] con la funzione di Beale (2D).
!  Questa funzione è progettata per mettere in crisi i modelli quadratici.
!  Punto di partenza: x = [0.1, 0.1] (in una zona molto piatta)
!  Minimo globale a x = [3.0, 0.5] dove f = 0.
!
   subroutine test_beale()

      implicit none

      integer, parameter :: wp = kind(1.0d0)
      integer, parameter :: n = 2
      real(wp), dimension(n) :: x, xl, xu
      integer :: npt, iprint, maxfun
      real(wp) :: rhobeg, rhoend

      ! Parametri per forzare la ricalibrazione del modello
      npt    = 2*n + 1        ! npt = 5
      iprint = 3              ! Massima verbosità per vedere i messaggi di ricalibrazione
      maxfun = 1000
      rhobeg = 0.5_wp         ! Raggio iniziale ampio per esplorare zone con curvatura variabile
      rhoend = 1.0e-8_wp      ! Tolleranza molto stretta

      ! Bounds ampi
      xl = -4.5_wp
      xu = 4.5_wp

      ! Punto iniziale in una zona piatta (plateau)
      x = (/0.1_wp, 0.1_wp/)

      write (*, *) '======================================================'
      write (*, *) 'BOBYQA STRESS TEST: Beale 2D Function'
      write (*, *) '======================================================'
      write (*, *) 'Dimensione (n): ', n
      write (*, *) 'Punti di interpolazione (npt): ', npt
      write (*, *) 'Punto iniziale: [0.1, 0.1]'
      write (*, *)

      ! Chiamata all'ottimizzatore BOBYQA
      call bobyqa(n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun, calfun_beale)

      write (*, *)
      write (*, *) '======================================================'
      write (*, *) 'Risultati Ottimizzazione:'
      write (*, *) 'Optimal x: ', x(1:min(n,5))
      write (*, *) 'Final f:   ', beale(x)
      write (*, *) '======================================================'

   contains

      !--------------------------------------------------------------------
      ! Funzione di Beale: f(x,y) = (1.5 - x + xy)^2 +
      !                            (2.25 - x + xy^2)^2 +
      !                            (2.625 - x + xy^3)^2
      !--------------------------------------------------------------------
      real(wp) function beale(x)
         real(wp), intent(in) :: x(:)
         real(wp) :: x1, x2, t1, t2, t3

         x1 = x(1)
         x2 = x(2)

         t1 = 1.5_wp - x1 + x1*x2
         t2 = 2.25_wp - x1 + x1*(x2**2)
         t3 = 2.625_wp - x1 + x1*(x2**3)

         beale = t1**2 + t2**2 + t3**2
      end function beale

      !--------------------------------------------------------------------
      ! Wrapper per BOBYQA
      !--------------------------------------------------------------------
      subroutine calfun_beale(n, x, f)
         integer, intent(in) :: n
         real(wp), intent(in) :: x(:)
         real(wp), intent(out) :: f

         f = beale(x)
      end subroutine calfun_beale

   end subroutine test_beale

subroutine test_flat_valley()

   use kind_module, only: wp
   implicit none

   integer, parameter :: n = 2
   real(wp) :: x(n), xl(n), xu(n)
   integer :: npt, iprint, maxfun
   real(wp) :: rhobeg, rhoend

   ! Configurazione aggressiva
   npt    = 2*n + 1      ! minimo: modello fragile
   iprint = 3            ! verbose (serve per vedere RESCUE)
   maxfun = 2000
   rhobeg = 1.0_wp
   rhoend = 1.0e-8_wp

   ! Bounds ampi
   xl = -20.0_wp
   xu =  20.0_wp

   ! Punto iniziale pessimo: fuori dalla valle
   x = (/ -10.0_wp, -10.0_wp /)

   write(*,*) '======================================================'
   write(*,*) 'BOBYQA STRESS TEST: Flat Valley (malcondizionata)'
   write(*,*) '======================================================'
   write(*,*) 'n    = ', n
   write(*,*) 'npt  = ', npt
   write(*,*) 'x0   = ', x
   write(*,*) 'rhobeg = ', rhobeg
   write(*,*) 'rhoend = ', rhoend
   write(*,*)

   call bobyqa(n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun, calfun_flat_valley)

   write(*,*)
   write(*,*) '------------------------------------------------------'
   write(*,*) 'Risultato finale'
   write(*,*) 'x* = ', x(1:min(5,m_nv))
   write(*,*) 'f* = ', flat_valley_eval(x)
   write(*,*) '======================================================'

contains

   real(wp) function flat_valley_eval(x)
      real(wp), intent(in) :: x(:)
      real(wp) :: a, b
      a = 1.0e-6_wp
      b = 1.0e+6_wp
      flat_valley_eval = a*(x(1)-1.0_wp)**2 + b*(x(2)-x(1)**2)**2
   end function
      !--------------------------------------------------------------------
      ! Wrapper per BOBYQA
      !--------------------------------------------------------------------
      subroutine calfun_flat_valley(n, x, f)
         integer, intent(in) :: n
         real(wp), intent(in) :: x(:)
         real(wp), intent(out) :: f

         f = flat_valley_eval(x)
      end subroutine calfun_flat_valley

end subroutine test_flat_valley


subroutine test_kinks()

   use kind_module, only: wp
   implicit none

   integer, parameter :: n = 3
   real(wp) :: x(n), xl(n), xu(n)
   integer :: npt, iprint, maxfun
   real(wp) :: rhobeg, rhoend, f

   ! Parametri deliberatamente cattivi
   npt    = 2*n + 1
   iprint = 3
   maxfun = 3000
   rhobeg = 1.5_wp
   rhoend = 1.0e-8_wp

   xl = -0.1_wp
   xu = 3.0_wp

   ! Punto iniziale vicino ai kink (abs)
   x = (/ -1, -1, 1 /)

   write(*,*) '======================================================'
   write(*,*) 'BOBYQA STRESS TEST: Kinks + Nonconvex Coupling'
   write(*,*) '======================================================'
   write(*,*) 'n    = ', n
   write(*,*) 'npt  = ', npt
   write(*,*) 'x0   = ', x
   write(*,*)

   call bobyqa(n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun, calfun_kinks)

   call calfun_kinks(n, x, f)

   write(*,*)
   write(*,*) '------------------------------------------------------'
   write(*,*) 'Risultato finale'
   write(*,*) 'x* = ', x(1:min(5,m_nv))
   write(*,*) 'f* = ', f
   write(*,*) '======================================================'

contains

subroutine calfun_kinks(n, x, f)
   use kind_module, only: wp
   integer, intent(in) :: n
   real(wp), intent(in) :: x(:)
   real(wp), intent(out) :: f

   integer :: i
   real(wp) :: r2

   f = 0.0_wp
   r2 = sum(x(1:n)**2)

   ! Plateau quasi piatto vicino all'origine
   if (r2 < 1.0_wp) then
      f = 1.0e-12_wp * r2
   else
      ! Salto improvviso di curvatura
      f = (r2 - 1.0_wp)**2
   end if

   ! Kinks non differenziabili
   do i = 1, n
      f = f + 1.0e-3_wp * abs(x(i))**1.3_wp
   end do

   ! Forte coupling tipo Rosenbrock
   do i = 1, n-1
      f = f + 50.0_wp * (x(i+1) - x(i)**2)**2
   end do
end subroutine calfun_kinks

end subroutine test_kinks




subroutine test_arglin()

   use kind_module, only: wp
   implicit none

   integer, parameter :: n = 50
   real(wp) :: x(n), xl(n), xu(n)
   integer :: npt, iprint, maxfun
   real(wp) :: rhobeg, rhoend, f

   ! Parametri deliberatamente cattivi
   npt    = (n+1)*(n+2)/2
   iprint = 3
   maxfun = 3000
   rhobeg = 1_wp
   rhoend = 1.0e-12_wp

   xl    = -10_wp
   xu    = 10_wp

   ! Punto iniziale vicino ai kink (abs)
   x = 1

   write(*,*) '======================================================'
   write(*,*) 'BOBYQA STRESS TEST: arglin'
   write(*,*) '======================================================'
   write(*,*) 'n    = ', n
   write(*,*) 'npt  = ', npt
   write(*,*) 'x0   = ', x
   write(*,*)

   call bobyqa(n, npt, x, xl, xu, rhobeg, rhoend, iprint, maxfun, calfun_arglin)

   call calfun_arglin(n, x, f)

   write(*,*)
   write(*,*) '------------------------------------------------------'
   write(*,*) 'Risultato finale'
   write(*,*) 'x* = ', x(1:min(5,m_nv))
   write(*,*) 'f* = ', f
   write(*,*) '======================================================'

contains

subroutine calfun_arglin(n, x, f)
   use kind_module, only: wp
   integer, intent(in) :: n
   real(wp), intent(in) :: x(:)
   real(wp), intent(out) :: f

   real(wp) :: ri
   integer :: i, j
   integer :: m

   ! Solitamente m = 2*n per i test standard
   m = 2 * n
   f = 0.0_wp

   do i = 1, m
      ri = 0.0_wp
      do j = 1, n
         ! Matrice del sistema: 1 sulla diagonale e nelle zone adiacenti
         if (i == j) then
            ri = ri + x(j)
         else
            ! Elementi fuori diagonale che creano dipendenza lineare
            ri = ri + 1.0e-4_wp * x(j)
         end if
      end do
      ! Sottraiamo 1 per spostare il minimo lontano dall'origine
      ri = ri - 1.0_wp
      f = f + ri**2
   end do

end subroutine calfun_arglin

end subroutine test_arglin


end module bobyqa_module
!*****************************************************************************************

!*****************************************************************************************
!>
!  Main program to run the BOBYQA test

program main
   use bobyqa_module
   implicit none

   write (*, *) 'Starting BOBYQA optimization test...'
   write (*, *) '====================================='

   call test_rosenbrock()

   call test_freudenstein_roth()

   call test_beale()

   call test_flat_valley()

   call test_kinks()

   call test_arglin()

   write (*, *) '====================================='
   write (*, *) 'Test completed.'

end program main
!*****************************************************************************************
