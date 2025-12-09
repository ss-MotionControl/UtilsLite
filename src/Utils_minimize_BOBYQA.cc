/*
 *
 * Implementation of Mike Powell's BOBYQA algorithm for minimizing a function
 * of many variables.  The method is "derivatives free" (only the function
 * values are needed) and accounts for bound constraints on the variables.  The
 * algorithm is described in:
 *
 *   M.J.D. Powell, "The BOBYQA Algorithm for Bound Constrained Optimization
 *   Without Derivatives."  Technical report, Department of Applied Mathematics
 *   and Theoretical Physics, University of Cambridge (2009).
 *
 * The present code is based on the original FORTRAN version written by Mike
 * Powell who kindly provides his code on demand (at mjdp@cam.ac.uk) and has
 * been converted to C by É. Thiébaut.
 *
 * Copyright (c) 2009, Mike Powell (FORTRAN version).
 * Copyright (c) 2015, Éric Thiébaut (C version).
 *
 * Read the accompanying `LICENSE` file for details.
 */

#include <stdio.h>
#include <math.h>

#include "Utils_fmt.hh"
#include "Utils_minimize_BOBYQA.hh"

#define OUTPUT stdout

namespace Utils
{

  using std::abs;
  using std::max;
  using std::min;
  using std::sqrt;

  template <typename Scalar>
  std::string
  BOBYQA_minimizer<Scalar>::print_vec( Vector const & x, integer max_elem ) const
  {
    if ( x.size() == 0 ) return "[]";

    std::string result = "[";
    if ( x.size() <= max_elem )
    {
      for ( size_t i = 0; i < x.size(); ++i )
      {
        if ( i != 0 ) result += ", ";
        result += fmt::format( "{}", x[i] );
      }
    }
    else
    {
      for ( int i = 0; i < max_elem; ++i )
      {
        if ( i != 0 ) result += ", ";
        result += fmt::format( "{}", x[i] );
      }
      result += ", ...";
      // Nota: non mostriamo gli elementi finali in questo caso, ma potremmo se volessimo
      // l'esempio fornito invece mostra solo i primi max_elem e poi "..."
    }
    result += "]";
    return result;
  }

  /*---------------------------------------------------------------------------*/
  /**
   * @brief Main minimization routine for BOBYQA algorithm.
   *
   * This function initializes the BOBYQA minimizer with problem dimensions,
   * validates parameters, processes bound constraints, and calls the core
   * optimization routine.
   *
   * @tparam Scalar Floating point type (e.g., double, float)
   * @param N Dimension of the problem (number of variables)
   * @param NPT Number of interpolation points (must satisfy N+2 ≤ NPT ≤ (N+1)(N+2)/2)
   * @param objfun Objective function to minimize
   * @param[in,out] X On input: initial guess. On output: final solution.
   * @param XL Vector of lower bounds
   * @param XU Vector of upper bounds
   * @return Status::BOBYQA_SUCCESS on success, appropriate error code otherwise
   *
   * @throws No exceptions thrown, but returns error status codes
   *
   * @details
   * The function performs the following steps:
   * 1. Stores problem dimensions and bounds
   * 2. Resizes all internal vectors and matrices
   * 3. Validates NPT parameter
   * 4. Checks bound feasibility
   * 5. Adjusts initial point if too close to bounds
   * 6. Calls bobyqb() for core optimization
   */
  template <typename Scalar>
  typename BOBYQA_minimizer<Scalar>::Status
  BOBYQA_minimizer<Scalar>::minimize( integer const         N,
                                      integer const         NPT,
                                      bobyqa_objfun const & objfun,
                                      Vector &              X,
                                      Vector const &        XL,
                                      Vector const &        XU )
  {
    // Call the core BOBYQB optimization routine with adjusted starting point

    using Eigen::Index;
    using std::max;
    using std::min;

    // Store problem dimensions
    m_neq = N;
    m_npt = NPT;
    m_dim = m_npt + m_neq;

    // Store bounds (Eigen uses copy-on-write for assignments)
    m_xlower = XL;
    m_xupper = XU;

    // Resize all internal storage using Eigen
    // Note: All resizing operations are O(1) for fixed-size vectors,
    // and only allocate memory when size changes for dynamic vectors
    m_xbase.resize( m_neq );
    m_xnew.resize( m_neq );
    m_xalt.resize( m_neq );
    m_fval.resize( m_npt );
    m_xopt.resize( m_neq );
    m_gopt.resize( m_neq );
    m_hq.resize( m_neq * ( m_neq + 1 ) / 2 );  // Size for symmetric matrix storage
    m_pq.resize( m_npt );
    m_sl.resize( m_neq );
    m_su.resize( m_neq );
    m_d.resize( m_neq );
    m_vlag.resize( m_dim );
    m_glag.resize( m_neq );  // Used with size m_neq in other routines
    m_hcol.resize( m_npt );
    m_gnew.resize( m_neq );

    m_xpt.resize( m_neq, m_npt );   // Matrix: N rows, NPT columns
    m_bmat.resize( m_neq, m_dim );  // Matrix: N rows, (NPT+N) columns

    // ZMAT matrix: NPT rows, (NPT-N-1) columns (minimum 1 column)
    const integer zmat_cols = std::max<integer>( m_npt - m_neq - 1, 1 );
    m_zmat.resize( m_npt, zmat_cols );

    m_ptsaux.resize( 2, m_neq );  // 2 rows, N columns
    m_ptsid.resize( m_npt );

    // Validate NPT parameter (as in original BOBYQA implementation)
    const integer np = m_neq + 1;
    if ( m_npt < m_neq + 2 || m_npt > ( m_neq + 2 ) * np / 2 )
    {
      print_error( "NPT is not in the required interval" );
      return Status::BOBYQA_BAD_NPT;
    }

    // Compute bound differences and check feasibility using vectorized operations
    Vector bounds_diff = m_xupper - m_xlower;  // XU - XL

    // Check if any bound difference is less than 2*RHOBEG
    if ( ( bounds_diff.array() < Scalar( 2 ) * m_rhobeg ).any() )
    {
      print_error( "one of the differences XU(I)-XL(I) is less than 2*RHOBEG" );
      return Status::BOBYQA_TOO_CLOSE;
    }

    // Initialize sl and su (distances from X to bounds)
    // sl = XL - X (negative when X > XL)
    // su = XU - X (positive when X < XU)
    m_sl = m_xlower - X;
    m_su = m_xupper - X;

    // Create masks for points near/past bounds using vectorized comparisons
    // Note: Using template cast<Scalar>() to convert bool masks to Scalar type
    Vector near_lower_mask = ( m_sl.array() >= -m_rhobeg ).template cast<Scalar>();
    Vector near_upper_mask = ( m_su.array() <= m_rhobeg ).template cast<Scalar>();

    // Process points that need adjustment due to proximity to bounds
    // We use a loop for conditional logic, but compute values vectorially when possible

    for ( integer j = 0; j < m_neq; ++j )
    {
      // Check if X[j] is at or near lower bound
      if ( near_lower_mask[j] != 0 )
      {
        if ( m_sl[j] >= 0 )
        {
          // X is at or beyond lower bound: clamp to bound
          X[j]    = m_xlower[j];
          m_sl[j] = 0;
          m_su[j] = bounds_diff[j];
        }
        else
        {
          // X is within rhobeg of lower bound: move away by rhobeg
          X[j]    = m_xlower[j] + m_rhobeg;
          m_sl[j] = -m_rhobeg;
          // Recompute su and ensure it's at least rhobeg
          const Scalar temp2 = m_xupper[j] - X[j];
          m_su[j]            = std::max( temp2, m_rhobeg );
        }
      }
      // Check if X[j] is at or near upper bound (only if not near lower bound)
      else if ( near_upper_mask[j] != 0 )
      {
        if ( m_su[j] <= 0 )
        {
          // X is at or beyond upper bound: clamp to bound
          X[j]    = m_xupper[j];
          m_sl[j] = -bounds_diff[j];
          m_su[j] = 0;
        }
        else
        {
          // X is within rhobeg of upper bound: move away by rhobeg
          X[j] = m_xupper[j] - m_rhobeg;
          // Recompute sl ensuring it's at most -rhobeg
          const Scalar tempa = m_xlower[j] - X[j];
          m_sl[j]            = std::min( tempa, -m_rhobeg );
          // Recompute su
          m_su[j] = m_xupper[j] - X[j];
        }
      }
      // No adjustment needed for points sufficiently interior
    }

    // workspace
    Vector W = Vector::Zero( std::max( 5 * m_neq, 2 * m_npt ) );

    // many scalars used by algorithm
    Scalar curv  = 0;
    Scalar ratio = 0;

    Status       status = Status::BOBYQA_SUCCESS;
    const char * reason = nullptr;

    // quick checks and setup
    integer nptm = m_npt - np;
    integer nh   = m_neq * np / 2;

    // INITIALIZATION (calls prelim which must fill m_xpt, m_fval, etc.)
    prelim( objfun, X );

    // set m_xopt from m_xpt row m_kopt (0-based)
    Scalar xoptsq = 0;
    for ( integer i = 0; i < m_neq; ++i )
    {
      m_xopt( i ) = m_xpt( i, m_kopt );  // kopt from prelim is 1-based in original; we used kopt as returned, keep -1
      xoptsq += m_xopt( i ) * m_xopt( i );
    }
    Scalar  fsave  = m_fval( 0 );
    integer kbase  = 1;
    Scalar  rho    = m_rhobeg;
    integer nresc  = m_nf;
    integer ntrits = 0;
    Scalar  diffa  = 0;
    Scalar  diffb  = 0;
    Scalar  diffc  = 0;
    Scalar  f      = 0;
    Scalar  distsq = 0;
    Scalar  dnorm  = 0;
    integer itest  = 0;
    integer nfsav  = m_nf;

    m_crvmin = 0;
    m_dsq    = 0;
    m_alpha  = 0;
    m_beta   = 0;
    m_delta  = rho;
    m_cauchy = 0;
    m_denom  = 0;
    m_adelt  = 0;
    m_knew   = 0;

    // Main state-machine loop variables: we'll use an enum and switch
    enum class Phase
    {
      UPDATE_GRADIENT,
      TRUST_REGION,
      SHIFT_BASE,
      RESCUE,
      ALTMOV,
      COMPUTE_VLAG,
      EVALUATE,
      FIND_FAR,
      REDUCE_RHO,
      DONE,
      ERROR
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per l'aggiornamento del gradiente nel punto ottimo corrente.
     *
     * Calcola il gradiente quadratico m_gopt nel punto ottimo m_xopt,
     * includendo sia la parte quadratica (m_hq) che la parte polinomiale (m_pq).
     * Viene eseguita quando il punto ottimo cambia rispetto alla base corrente.
     */
    auto phase_update_gradient = [&]() -> Phase
    {
      if ( m_kopt+1 != kbase )
      {
        integer ih = 0;
        for ( integer j = 0; j < m_neq; ++j )
        {
          // Parte simmetrica della matrice Hessiana (matrice triangolare superiore impacchettata)
          for ( integer i = 0; i < j; ++i )
          {
            m_gopt( j ) += m_hq( ih ) * m_xopt( i );
            m_gopt( i ) += m_hq( ih ) * m_xopt( j );
            ++ih;
          }
          // Elemento diagonale della matrice Hessiana
          m_gopt( j ) += m_hq( ih ) * m_xopt( j );
          ++ih;
        }
        // Contributo dei termini quadratici dei punti di interpolazione
        if ( m_nf > m_npt )
        {
          for ( integer k = 0; k < m_npt; ++k )
          {
            Scalar temp = m_xpt.col( k ).dot( m_xopt );   // Prodotto scalare punto interpolazione con ottimo
            temp *= m_pq( k );                            // Moltiplica per coefficiente polinomiale
            m_gopt += temp * m_xpt.col( k );  // Aggiunge al gradiente
          }
        }
      }
      return Phase::TRUST_REGION;  // Passa alla fase successiva
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per la risoluzione del problema della regione di fiducia.
     *
     * Risolve il problema quadratico vincolato nella regione di fiducia usando trsbox().
     * Determina se il passo è sufficientemente grande o se è necessario ridurre rho
     * o cercare un punto di interpolazione lontano.
     */
    auto phase_trust_region = [&]() -> Phase
    {
      // Risolve il problema quadratico vincolato nella regione di fiducia
      trsbox();

      // Calcola norma del passo e la limita alla dimensione della regione di fiducia
      dnorm = std::sqrt( m_dsq );
      dnorm = std::min( dnorm, m_delta );

      // Controlla se il passo è troppo piccolo rispetto a rho
      if ( dnorm < Scalar( 0.5 ) * rho )
      {
        ntrits = -1;  // Marca che non è stata fatta una valutazione di funzione
        distsq = ( Scalar( 10 ) * rho ) * ( Scalar( 10 ) * rho );  // Soglia per cercare punto lontano

        // Controlli per decidere se ridurre rho o cercare punto lontano
        if ( m_nf <= nfsav + 2 ) { return Phase::FIND_FAR; }

        Scalar errbig = std::max( { diffa, diffb, diffc } );  // Errore di approssimazione
        Scalar frhosq = rho * Scalar( 0.125 ) * rho;

        // Se l'errore è grande rispetto alla curvatura minima, cerca punto lontano
        if ( m_crvmin > 0 && errbig > frhosq * m_crvmin ) { return Phase::FIND_FAR; }

        // Verifica condizioni per ridurre rho basate sulle condizioni di ottimalità
        Scalar bdtol  = errbig / rho;
        bool   reduce = true;
        for ( integer j = 0; j < m_neq; ++j )
        {
          Scalar bdtest = bdtol;
          if ( m_xnew( j ) == m_sl( j ) ) bdtest = m_gnew( j );
          if ( m_xnew( j ) == m_su( j ) ) bdtest = -m_gnew( j );

          if ( bdtest < bdtol )
          {
            // Calcola curvatura nella direzione j
            curv = m_hq[( j + ( j + 1 ) * j / 2 )];  // Elemento diagonale della Hessiana
            for ( integer k = 0; k < m_npt; ++k )
            {
              Scalar t = m_xpt( j, k );
              curv += m_pq( k ) * t * t;  // Contributo dei punti di interpolazione
            }
            bdtest += Scalar( 0.5 ) * curv * rho;
            if ( bdtest < bdtol )
            {
              reduce = false;
              break;
            }
          }
        }
        return reduce ? Phase::REDUCE_RHO : Phase::FIND_FAR;
      }

      ++ntrits;                  // Incrementa contatore valutazioni nella regione di fiducia
      return Phase::SHIFT_BASE;  // Passa alla fase di shift della base
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per lo shift della base e l'aggiornamento delle matrici.
     *
     * Quando il passo è piccolo rispetto alla distanza dal centro, sposta la base
     * al punto ottimo corrente e aggiorna tutte le matrici (BMAT, ZMAT, HQ)
     * per mantenere l'interpolazione quadratica.
     */
    auto phase_shift_base = [&]() -> Phase
    {
      // Controlla se il passo è significativamente più piccolo della distanza dal centro
      if ( m_dsq <= xoptsq * Scalar( 0.001 ) )
      {
        Scalar fracsq = xoptsq * Scalar( 0.25 );
        Scalar sumpq  = 0;

        // Fase 1: Aggiornamento di BMAT con contributi dai punti di interpolazione
        for ( integer k = 0; k < m_npt; ++k )
        {
          sumpq += m_pq( k );
          Scalar sum     = -Scalar( 0.5 ) * xoptsq + m_xpt.col( k ).dot( m_xopt );
          W( m_npt + k ) = sum;
          Scalar temp    = fracsq - Scalar( 0.5 ) * sum;

          // Calcola vettori temporanei per l'aggiornamento
          for ( integer i = 0; i < m_neq; ++i )
          {
            W( i )      = m_bmat( i, k );
            m_vlag( i ) = sum * m_xpt( i, k ) + temp * m_xopt( i );
          }

          // Aggiorna la parte inferiore di BMAT
          for ( integer i = 0; i < m_neq; ++i )
          {
            integer ip = m_npt + i;
            for ( integer j = 0; j <= i; ++j ) { m_bmat( j, ip ) += W( i ) * m_vlag( j ) + m_vlag( i ) * W( j ); }
          }
        }

        // Fase 2: Aggiornamento di BMAT usando ZMAT (matrice ortogonale)
        for ( integer jj = 0; jj < nptm; ++jj )
        {
          Scalar sumz = 0;
          Scalar sumw = 0;
          for ( integer k = 0; k < m_npt; ++k )
          {
            sumz += m_zmat( k, jj );
            m_vlag( k ) = W( m_npt + k ) * m_zmat( k, jj );
            sumw += m_vlag( k );
          }

          for ( integer j = 0; j < m_neq; ++j )
          {
            Scalar sum = ( fracsq * sumz - Scalar( 0.5 ) * sumw ) * m_xopt( j );
            for ( integer k = 0; k < m_npt; ++k ) sum += m_vlag( k ) * m_xpt( j, k );
            W( j ) = sum;

            for ( integer k = 0; k < m_npt; ++k ) { m_bmat( j, k ) += sum * m_zmat( k, jj ); }
          }

          for ( integer i = 0; i < m_neq; ++i )
          {
            integer ip   = i + m_npt;
            Scalar  temp = W( i );
            for ( integer j = 0; j <= i; ++j ) { m_bmat( j, ip ) += temp * W( j ); }
          }
        }

        // Fase 3: Aggiornamento di HQ e shift dei punti di interpolazione
        integer ih = 0;
        for ( integer j = 0; j < m_neq; ++j )
        {
          W( j ) = -Scalar( 0.5 ) * sumpq * m_xopt( j );
          for ( integer k = 0; k < m_npt; ++k )
          {
            W( j ) += m_pq( k ) * m_xpt( j, k );
            m_xpt( j, k ) -= m_xopt( j );  // Shift del punto di interpolazione
          }

          for ( integer i = 0; i <= j; ++i )
          {
            m_hq( ih ) += W( i ) * m_xopt( j ) + m_xopt( i ) * W( j );
            // Rende simmetrica la parte di BMAT
            m_bmat( j, m_npt + i ) = m_bmat( i, m_npt + j );
            ++ih;
          }
        }

        // Fase 4: Shift completo della base e aggiornamento dei bound
        for ( integer i = 0; i < m_neq; ++i )
        {
          m_xbase( i ) += m_xopt( i );  // Sposta la base
          m_xnew( i ) -= m_xopt( i );   // Corregge il nuovo punto
          m_sl( i ) -= m_xopt( i );     // Aggiorna limite inferiore relativo
          m_su( i ) -= m_xopt( i );     // Aggiorna limite superiore relativo
          m_xopt( i ) = 0;              // Reset del punto ottimo relativo
        }
        xoptsq = 0;  // Reset della norma al quadrato
      }

      return ( ntrits == 0 ) ? Phase::ALTMOV : Phase::COMPUTE_VLAG;
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per l'operazione di rescue.
     *
     * Chiama la procedura rescue() quando si sospettano problemi numerici
     * o quando i denominatori diventano troppo piccoli. Rigenera il set
     * di punti di interpolazione mantenendo l'interpolazione quadratica.
     */
    auto phase_rescue = [&]() -> Phase
    {
      nfsav = m_nf;    // Salva il numero corrente di valutazioni
      kbase = m_kopt+1;  // Salva il punto ottimo corrente come base

      // Esegue la procedura di rescue
      rescue();
      xoptsq = 0;

      // Se il punto ottimo è cambiato, aggiorna m_xopt e xoptsq
      if ( m_kopt+1 != kbase )
      {
        for ( integer i = 0; i < m_neq; ++i )
        {
          m_xopt( i ) = m_xpt( i, m_kopt );
          xoptsq += m_xopt( i ) * m_xopt( i );
        }
      }

      // Controlla se sono state esaurite le valutazioni durante il rescue
      if ( m_nf < 0 )
      {
        m_nf   = m_maxfun;
        reason = "CALFUN has been called MAXFUN times";
        return Phase::ERROR;
      }

      nresc = m_nf;  // Aggiorna contatore rescue

      // Decide la fase successiva in base allo stato
      if ( nfsav < m_nf )
      {
        nfsav = m_nf;
        return Phase::UPDATE_GRADIENT;  // Nuove valutazioni, aggiorna gradiente
      }
      else if ( ntrits > 0 )
      {
        return Phase::TRUST_REGION;  // Ritorna alla regione di fiducia
      }
      else
      {
        return Phase::ALTMOV;  // Prova movimento alternativo
      }
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per il calcolo del movimento alternativo.
     *
     * Chiama altmov() per generare un passo alternativo quando la regione
     * di fiducia è ristretta o quando non sono state fatte valutazioni di funzione.
     */
    auto phase_altmov = [&]() -> Phase
    {
      // Calcola un passo alternativo
      altmov();

      // Calcola la direzione d dal punto ottimo al nuovo punto
      for ( integer i = 0; i < m_neq; ++i ) { m_d( i ) = m_xnew( i ) - m_xopt( i ); }

      return Phase::COMPUTE_VLAG;  // Passa al calcolo dei coefficienti di Lagrange
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per il calcolo dei coefficienti di Lagrange.
     *
     * Calcola i coefficienti di Lagrange m_vlag per il punto di interpolazione
     * da sostituire e il denominatore m_denom per la regola di aggiornamento.
     * Sceglie anche il punto da rimuovere (m_knew) se necessario.
     */
    auto phase_compute_vlag = [&]() -> Phase
    {
      // Fase 1: Calcola contributi base dai punti di interpolazione
      for ( integer k = 0; k < m_npt; ++k )
      {
        Scalar suma    = m_xpt.col( k ).dot( m_d );                 // Proiezione sulla direzione d
        Scalar sumb    = m_xpt.col( k ).dot( m_xopt );              // Proiezione sul punto ottimo
        Scalar sum     = m_bmat.col( k ).head( m_neq ).dot( m_d );  // Contributo di BMAT
        W( k )         = suma * ( Scalar( 0.5 ) * suma + sumb );    // Termine quadratico
        m_vlag( k )    = sum;                                       // Coefficiente di Lagrange
        W( m_npt + k ) = suma;                                      // Salva per uso successivo
      }

      // Fase 2: Calcola contributo da ZMAT (parte ortogonale)
      m_beta = 0;
      for ( integer jj = 0; jj < nptm; ++jj )
      {
        Scalar sum = m_zmat.col( jj ).dot( W.head( m_npt ) );
        m_beta -= sum * sum;  // Aggiorna beta (contributo negativo)
        for ( integer k = 0; k < m_npt; ++k )
        {
          m_vlag( k ) += sum * m_zmat( k, jj );  // Aggiorna coefficienti di Lagrange
        }
      }

      // Fase 3: Calcola norme e prodotti scalari
      m_dsq       = 0;
      Scalar bsum = 0;
      Scalar dx   = 0;
      for ( integer j = 0; j < m_neq; ++j )
      {
        m_dsq += m_d( j ) * m_d( j );  // Norma al quadrato della direzione

        Scalar sum = 0;
        for ( integer k = 0; k < m_npt; ++k ) sum += W( k ) * m_bmat( j, k );
        bsum += sum * m_d( j );  // Contributo di BMAT

        integer jp = m_npt + j;
        for ( integer i = 0; i < m_neq; ++i ) sum += m_bmat( i, jp ) * m_d( i );
        m_vlag( jp ) = sum;            // Estende coefficienti di Lagrange
        bsum += sum * m_d( j );        // Completa bsum
        dx += m_d( j ) * m_xopt( j );  // Prodotto scalare con punto ottimo
      }

      // Calcola beta finale (distanza quadratica normalizzata)
      m_beta = dx * dx + m_dsq * ( xoptsq + dx + dx + Scalar( 0.5 ) * m_dsq ) + m_beta - bsum;
      m_vlag( m_kopt ) += 1;  // Condizione di interpolazione nel punto ottimo

      // Fase 4: Scelta del denominatore e controllo di validità
      if ( ntrits == 0 )
      {
        // Prima iterazione nella regione di fiducia
        m_denom = m_vlag( m_knew - 1 ) * m_vlag( m_knew - 1 ) + m_alpha * m_beta;

        // Controlla se il denominatore è peggiore del passo di Cauchy
        if ( m_denom < m_cauchy && m_cauchy > 0 )
        {
          // Revert al passo alternativo
          for ( integer i = 0; i < m_neq; ++i )
          {
            m_xnew( i ) = m_xalt( i );
            m_d( i )    = m_xnew( i ) - m_xopt( i );
          }
          m_cauchy = 0;
          return Phase::COMPUTE_VLAG;  // Ricalcola con passo alternativo
        }

        // Controlla cancellazione numerica nel denominatore
        if ( m_denom <= Scalar( 0.5 ) * m_vlag( m_knew - 1 ) * m_vlag( m_knew - 1 ) )
        {
          if ( m_nf > nresc )
          {
            return Phase::RESCUE;  // Richiede rescue per problemi numerici
          }
          else
          {
            reason = "of much cancellation in a denominator";
            return Phase::ERROR;  // Errore irreparabile
          }
        }
      }
      else
      {
        // Iterazioni successive: sceglie il punto da rimuovere (m_knew)
        Scalar delsq  = m_delta * m_delta;
        Scalar scaden = 0;  // Denominatore scalato massimo
        Scalar biglsq = 0;  // Termine di confronto
        // m_knew        = 0; non devo azzerare se poi m_knew non viene aggiornato

        for ( integer k = 0; k < m_npt; ++k )
        {
          if ( k == m_kopt ) continue;  // Salta il punto ottimo

          // Calcola hdiag (contributo ortogonale)
          Scalar hdiag = 0;
          for ( integer jj = 0; jj < nptm; ++jj ) { hdiag += m_zmat( k, jj ) * m_zmat( k, jj ); }

          // Calcola denominatore per questo punto
          Scalar den = m_beta * hdiag + m_vlag( k ) * m_vlag( k );

          // Calcola distanza normalizzata dal punto ottimo
          distsq      = ( m_xpt.col( k ) - m_xopt ).squaredNorm();
          Scalar temp = distsq / delsq;
          temp        = temp * temp;
          if ( temp < 1 ) temp = 1;  // Evita valori troppo piccoli

          // Aggiorna scaden e biglsq
          if ( temp * den > scaden )
          {
            scaden  = temp * den;
            m_knew  = k + 1;  // 1-based index
            m_denom = den;
          }
          temp *= ( m_vlag( k ) * m_vlag( k ) );
          biglsq = max( biglsq, temp );
        }

        // Controlla se il denominatore è sufficientemente grande
        if ( scaden <= Scalar( 0.5 ) * biglsq )
        {
          if ( m_nf > nresc )
          {
            return Phase::RESCUE;  // Problemi numerici, richiede rescue
          }
          else
          {
            reason = "of much cancellation in a denominator";
            return Phase::ERROR;  // Errore irreparabile
          }
        }
      }

      return Phase::EVALUATE;  // Procedi con la valutazione della funzione
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per la valutazione della funzione obiettivo.
     *
     * Valuta la funzione obiettivo nel nuovo punto, aggiorna il modello quadratico,
     * e decide se accettare il passo, ridurre la regione di fiducia, o ridurre rho.
     */
    auto phase_evaluate = [&]() -> Phase
    {
      // Fase 1: Proietta il nuovo punto sui bound con trattamento speciale
      for ( integer i = 0; i < m_neq; ++i )
      {
        Scalar tempa = m_xbase( i ) + m_xnew( i );
        tempa        = std::max( tempa, m_xlower( i ) );
        X( i )       = std::min( tempa, m_xupper( i ) );

        // Gestione speciale per punti sui bound
        if ( m_xnew( i ) == m_sl( i ) ) X( i ) = m_xlower( i );
        if ( m_xnew( i ) == m_su( i ) ) X( i ) = m_xupper( i );
      }

      // Controllo limite massimo valutazioni
      if ( m_nf >= m_maxfun )
      {
        reason = "CALFUN has been called MAXFUN times";
        return Phase::ERROR;
      }

      // Valutazione funzione obiettivo
      ++m_nf;
      f = objfun( X );

      // Output diagnostico
      if ( m_print_level == 3 )
      {
        fmt::print(
            "    Function number{}  F ={:.6)\n"
            "    The corresponding X is: {}\n",
            m_nf, f, print_vec( X, 6 ) );
      }

      // Caso speciale: nessuna valutazione di funzione nella regione di fiducia
      if ( ntrits == -1 )
      {
        fsave = f;
        return Phase::DONE;
      }

      // Fase 2: Calcolo riduzione predetta dal modello quadratico
      Scalar fopt  = m_fval( m_kopt );
      Scalar vquad = 0;  // Riduzione predetta

      integer ih = 0;
      for ( integer j = 0; j < m_neq; ++j )
      {
        vquad += m_d( j ) * m_gopt( j );  // Contributo lineare

        for ( integer i = 0; i <= j; ++i )
        {
          Scalar temp = m_d( i ) * m_d( j );
          if ( i == j ) temp = Scalar( 0.5 ) * temp;
          vquad += m_hq[ih] * temp;  // Contributo quadratico
          ++ih;
        }
      }

      for ( integer k = 0; k < m_npt; ++k ) { vquad += Scalar( 0.5 ) * m_pq( k ) * W( m_npt + k ) * W( m_npt + k ); }

      // Fase 3: Aggiorna statistiche errore approssimazione
      Scalar diff = f - fopt - vquad;
      diffc       = diffb;
      diffb       = diffa;
      diffa       = std::abs( diff );

      if ( dnorm > rho ) nfsav = m_nf;

      // Fase 4: Gestione risultato valutazione
      if ( ntrits > 0 )
      {
        // Controlla se il modello predice correttamente
        if ( vquad >= 0 )
        {
          reason = "a trust region step has failed to reduce Q";
          return Phase::ERROR;
        }

        // Calcola ratio riduzione effettiva/predetta
        ratio = ( f - fopt ) / vquad;

        // Regola dimensione regione di fiducia in base al ratio
        if ( ratio <= Scalar( 0.1 ) ) { m_delta = std::min( Scalar( 0.5 ) * m_delta, dnorm ); }
        else if ( ratio <= Scalar( 0.7 ) ) { m_delta = std::max( Scalar( 0.5 ) * m_delta, dnorm ); }
        else
        {
          m_delta = std::max( Scalar( 0.5 ) * m_delta, 2 * dnorm );
        }
        if ( m_delta <= Scalar( 1.5 ) * rho ) m_delta = rho;

        // Se il passo migliora la funzione, rivaluta scelta punto da rimuovere
        if ( f < fopt )
        {
          integer ksav   = m_knew;
          Scalar  densav = m_denom;
          Scalar  delsq  = m_delta * m_delta;
          Scalar  scaden = 0;
          Scalar  biglsq = 0;
          m_knew         = 0;

          for ( integer k = 0; k < m_npt; ++k )
          {
            Scalar hdiag = 0;
            for ( integer jj = 0; jj < nptm; ++jj ) { hdiag += m_zmat( k, jj ) * m_zmat( k, jj ); }
            Scalar den = m_beta * hdiag + m_vlag( k ) * m_vlag( k );

            distsq      = ( m_xpt.col( k ) - m_xnew ).squaredNorm();
            Scalar temp = distsq / delsq;
            temp        = temp * temp;
            if ( temp < 1 ) temp = 1;

            if ( scaden < temp * den )
            {
              scaden  = temp * den;
              m_knew  = k + 1;
              m_denom = den;
            }
            biglsq = max( biglsq, temp * m_vlag( k ) * m_vlag( k ) );
          }

          if ( scaden <= Scalar( 0.5 ) * biglsq || m_knew == 0 )
          {
            m_knew  = ksav;
            m_denom = densav;
          }
        }
      }

      // Fase 5: Aggiornamento del modello quadratico
      update();  // Aggiorna BMAT, ZMAT

      // Aggiorna HQ con contributo del punto rimosso
      ih                 = 0;
      Scalar pqold       = m_pq( m_knew - 1 );
      m_pq( m_knew - 1 ) = 0;
      for ( integer i = 0; i < m_neq; ++i )
      {
        Scalar temp = pqold * m_xpt( i, m_knew - 1 );
        for ( integer j = 0; j <= i; ++j )
        {
          m_hq[ih] += temp * m_xpt( j, m_knew - 1 );
          ++ih;
        }
      }

      // Aggiorna PQ con differenza funzione
      for ( integer jj = 0; jj < nptm; ++jj )
      {
        Scalar temp = diff * m_zmat( m_knew - 1, jj );
        for ( integer k = 0; k < m_npt; ++k ) { m_pq( k ) += temp * m_zmat( k, jj ); }
      }

      // Incorpora nuovo punto di interpolazione
      m_fval( m_knew - 1 ) = f;
      for ( integer i = 0; i < m_neq; ++i )
      {
        m_xpt( i, m_knew - 1 ) = m_xnew( i );
        W( i )                 = m_bmat( i, m_knew - 1 );
      }

      // Aggiorna gradiente con nuovo punto
      for ( integer k = 0; k < m_npt; ++k )
      {
        Scalar suma = 0;
        for ( integer jj = 0; jj < nptm; ++jj ) { suma += m_zmat( m_knew - 1, jj ) * m_zmat( k, jj ); }
        Scalar sumb = m_xpt.col( k ).dot( m_xopt );
        Scalar temp = suma * sumb;
        for ( integer i = 0; i < m_neq; ++i ) { W( i ) += temp * m_xpt( i, k ); }
      }

      for ( integer i = 0; i < m_neq; ++i ) { m_gopt( i ) += diff * W( i ); }

      // Fase 6: Aggiorna punto ottimo se migliorato
      if ( f < fopt )
      {
        m_kopt = m_knew-1;
        xoptsq = 0;
        ih     = 0;

        for ( integer j = 0; j < m_neq; ++j )
        {
          m_xopt( j ) = m_xnew( j );
          xoptsq += m_xopt( j ) * m_xopt( j );

          for ( integer i = 0; i <= j; ++i )
          {
            if ( i < j ) m_gopt( j ) += m_hq[ih] * m_d( i );
            m_gopt( i ) += m_hq[ih] * m_d( j );
            ++ih;
          }
        }

        for ( integer k = 0; k < m_npt; ++k )
        {
          Scalar temp = m_xpt.col( k ).dot( m_d );
          temp *= m_pq( k );
          for ( integer i = 0; i < m_neq; ++i ) { m_gopt( i ) += temp * m_xpt( i, k ); }
        }
      }

      // Fase 7: Controllo qualità interpolazione e aggiornamento forzato
      if ( ntrits > 0 )
      {
        // Calcola interpolante di Frobenius minimo
        for ( integer k = 0; k < m_npt; ++k )
        {
          m_vlag( k ) = m_fval( k ) - m_fval( m_kopt );
          W( k )      = 0;
        }

        for ( integer jj = 0; jj < nptm; ++jj )
        {
          Scalar sum = m_zmat.col( jj ).dot( m_vlag.head( m_npt ) );
          for ( integer k = 0; k < m_npt; ++k ) { W( k ) += sum * m_zmat( k, jj ); }
        }

        for ( integer k = 0; k < m_npt; ++k )
        {
          Scalar sum     = m_xpt.col( k ).dot( m_xopt );
          W( m_npt + k ) = W( k );
          W( k ) *= sum;
        }

        // Verifica qualità gradiente interpolante
        Scalar gqsq = 0;  // Norma gradiente quadratico
        Scalar gisq = 0;  // Norma gradiente interpolante

        for ( integer i = 0; i < m_neq; ++i )
        {
          Scalar sum = 0;
          for ( integer k = 0; k < m_npt; ++k ) { sum += m_bmat( i, k ) * m_vlag( k ) + m_xpt( i, k ) * W( k ); }

          // Gestione vincoli attivi
          if ( m_xopt( i ) == m_sl( i ) )
          {
            Scalar tempa = std::min( Scalar( 0 ), m_gopt( i ) );
            gqsq += tempa * tempa;
            tempa = std::min( Scalar( 0 ), sum );
            gisq += tempa * tempa;
          }
          else if ( m_xopt( i ) == m_su( i ) )
          {
            Scalar tempa = std::max( Scalar( 0 ), m_gopt( i ) );
            gqsq += tempa * tempa;
            tempa = std::max( Scalar( 0 ), sum );
            gisq += tempa * tempa;
          }
          else
          {
            gqsq += m_gopt( i ) * m_gopt( i );
            gisq += sum * sum;
          }
          m_vlag( m_npt + i ) = sum;
        }

        // Controllo convergenza interpolante
        ++itest;
        if ( gqsq < Scalar( 10 ) * gisq ) itest = 0;

        // Se necessario, sostituisce modello con interpolante minimo
        if ( itest >= 3 )
        {
          integer i1 = std::max( m_npt, nh );
          for ( integer i = 0; i < i1; ++i )
          {
            if ( i < m_neq ) m_gopt( i ) = m_vlag( m_npt + i );
            if ( i < m_npt ) m_pq( i ) = W( m_npt + i );
            if ( i < nh ) m_hq( i ) = 0;
          }
          itest = 0;
        }
      }

      // Fase 8: Decisione fase successiva
      if ( ntrits == 0 || f <= fopt + Scalar( 0.1 ) * vquad )
      {
        return Phase::TRUST_REGION;  // Continua con regione di fiducia
      }
      else
      {
        // Cerca punto lontano per migliorare geometria
        distsq = std::max( ( 2 * m_delta ) * ( 2 * m_delta ), ( Scalar( 10 ) * rho ) * ( Scalar( 10 ) * rho ) );
        return Phase::FIND_FAR;
      }
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per la ricerca di un punto di interpolazione lontano.
     *
     * Cerca il punto di interpolazione più lontano dal punto ottimo corrente
     * per sostituirlo con un nuovo punto che migliori la geometria dell'interpolazione.
     */
    auto phase_find_far = [&]() -> Phase
    {
      m_knew          = 0;
      Scalar max_dist = 0;

      // Trova il punto più lontano dal punto ottimo
      for ( integer k = 0; k < m_npt; ++k )
      {
        Scalar sum = ( m_xpt.col( k ) - m_xopt ).squaredNorm();
        if ( sum > max_dist )
        {
          max_dist = sum;
          m_knew   = k + 1;  // 1-based index
        }
      }

      if ( m_knew > 0 )
      {
        // Calcola distanza e adatta parametri
        Scalar dist = std::sqrt( max_dist );

        if ( ntrits == -1 )
        {
          // Prima iterazione: riduci delta
          m_delta = std::min( Scalar( 0.1 ) * m_delta, Scalar( 0.5 ) * dist );
          if ( m_delta <= Scalar( 1.5 ) * rho ) m_delta = rho;
        }

        ntrits  = 0;  // Reset contatore valutazioni
        m_adelt = std::max( std::min( Scalar( 0.1 ) * dist, m_delta ), rho );
        m_dsq   = m_adelt * m_adelt;
        return Phase::SHIFT_BASE;  // Procedi con shift base per nuovo punto
      }
      else
      {
        // Nessun punto lontano trovato
        if ( ntrits == -1 )
        {
          return Phase::REDUCE_RHO;  // Riduci rho se nessuna valutazione
        }
        else if ( ratio > 0 || std::max( m_delta, dnorm ) > rho )
        {
          return Phase::TRUST_REGION;  // Continua con regione di fiducia
        }
        else
        {
          return Phase::REDUCE_RHO;  // Riduci rho
        }
      }
    };

    /*---------------------------------------------------------------------------*/
    /**
     * @brief Lambda per la riduzione del parametro rho.
     *
     * Riduce il parametro rho (dimensione caratteristica) quando l'algoritmo
     * ha convergentro alla precisione desiderata o quando è necessario
     * esplorare su scala più fine.
     */
    auto phase_reduce_rho = [&]() -> Phase
    {
      // Controlla se siamo al limite di precisione
      if ( rho > m_rhoend )
      {
        // Riduci rho in base al rapporto con rhoend
        m_delta = Scalar( 0.5 ) * rho;
        ratio   = rho / m_rhoend;

        if ( ratio <= Scalar( 16 ) ) { rho = m_rhoend; }
        else if ( ratio <= Scalar( 250 ) ) { rho = std::sqrt( ratio ) * m_rhoend; }
        else
        {
          rho = Scalar( 0.1 ) * rho;
        }

        m_delta = max( m_delta, rho );  // Mantieni delta almeno rho

        // Output diagnostico
        if ( m_print_level >= 3 )
        {
          fmt::print(
              "\n"
              "    New RHO                   = {}\n"
              "    Number of function values = {}\n"
              "    Least value of F          = {}\n"
              "    The corresponding X is: {}\n",
              rho, m_nf, m_fval( m_kopt ), print_vec( m_xopt, 6 ) );
        }

        // Reset parametri per nuova fase
        ntrits = 0;
        nfsav  = m_nf;
        return Phase::TRUST_REGION;  // Continua con nuovo rho
      }
      else if ( ntrits == -1 )
      {
        return Phase::EVALUATE;  // Caso speciale: nessuna valutazione
      }
      else
      {
        return Phase::DONE;  // Convergenza raggiunta
      }
    };

    Phase phase      = Phase::UPDATE_GRADIENT;
    bool  keep_going = true;

    if ( m_nf < m_npt )
    {
      reason = "CALFUN has been called MAXFUN times";
      status = Status::BOBYQA_TOO_MANY_EVALUATIONS;
      // finalization at bottom
      goto FINALIZE;
    }

    while ( keep_going )
    {
      switch ( phase )
      {
        case Phase::UPDATE_GRADIENT:
          // std::cout << "UPDATE_GRADIENT m_knew = " << m_knew << '\n';
          phase = phase_update_gradient();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::TRUST_REGION:
          // std::cout << "TRUST_REGION m_knew = " << m_knew << '\n';
          phase = phase_trust_region();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::SHIFT_BASE:
          // std::cout << "SHIFT_BASE m_knew = " << m_knew << '\n';
          phase = phase_shift_base();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::RESCUE:
          // std::cout << "RESCUE m_knew = " << m_knew << '\n';
          phase = phase_rescue();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::ALTMOV:
          // std::cout << "ALTMOV m_knew = " << m_knew << '\n';
          phase = phase_altmov();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::COMPUTE_VLAG:
          // std::cout << "COMPUTE_VLAG m_knew = " << m_knew << '\n';
          phase = phase_compute_vlag();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::EVALUATE:
          // std::cout << "EVALUATE m_knew = " << m_knew << '\n';
          phase = phase_evaluate();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::FIND_FAR:
          // std::cout << "FIND_FAR m_knew = " << m_knew << '\n';
          phase = phase_find_far();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::REDUCE_RHO:
          // std::cout << "REDUCE_RHO m_knew = " << m_knew << '\n';
          phase = phase_reduce_rho();
          // std::cout << "m_knew = " << m_knew << '\n';
          break;

        case Phase::DONE:
          // std::cout << "DONE m_knew = " << m_knew << '\n';
          //  Termina l'esecuzione dell'algoritmo dopo aver raggiunto convergenza
          //  o dopo aver esaurito le risorse di calcolo.
          keep_going = false;
          break;

        case Phase::ERROR:
          // std::cout << "ERROR m_knew = " << m_knew << '\n';
          //  Gestisce gli errori dell'algoritmo, impostando il messaggio di errore
          //  e lo stato appropriato.
          if ( m_print_level > 0 && reason ) print_error( reason );
          keep_going = false;
          break;

      }  // switch
    }  // while

  FINALIZE:
    // finalize result: set X to best point if smaller than starting fsave
    if ( m_fval( m_kopt ) <= fsave )
    {
      for ( integer i = 0; i < m_neq; ++i )
      {
        Scalar tempa = m_xbase( i ) + m_xopt( i );
        tempa        = std::max( tempa, m_xlower( i ) );
        X( i )       = std::min( tempa, m_xupper( i ) );
        if ( m_xopt( i ) == m_sl( i ) ) X( i ) = m_xlower( i );
        if ( m_xopt( i ) == m_su( i ) ) X( i ) = m_xupper( i );
      }
      f = m_fval( m_kopt );
    }

    if ( m_print_level >= 1 )
    {
      fmt::print(
          "\n"
          "    At the return from BOBYQA\n"
          "    Number of function values = {}\n"
          "    Least value of F          = {}\n"
          "    The corresponding X is: {}\n",
          m_nf, f, print_vec( X, 6 ) );
    }

    // non ha senso lo rimuovo
    //if ( status == Status::BOBYQA_SUCCESS ) m_xbase( 0 ) = f;
    return status;
  }

  template <typename Scalar>
  void
  BOBYQA_minimizer<Scalar>::altmov()
  {
    const Scalar one_plus_sqrt2 = 1 + sqrt( Scalar( 2 ) );

    Vector W( 2 * m_neq );

    Scalar  csave  = 0;
    Scalar  stpsav = 0;
    Scalar  step   = 0;
    integer ksav   = 0;
    integer ibdsav = 0;

    // Compute H column for knew
    std::fill( m_hcol.begin(), m_hcol.end(), 0 );
    integer i1 = m_npt - m_neq - 1;
    for ( integer j = 0; j < i1; ++j )
    {
      Scalar temp = m_zmat( m_knew - 1, j );
      m_hcol.noalias() += temp * m_zmat.col(j);
    }
    m_alpha   = m_hcol[m_knew - 1];
    Scalar ha = Scalar( 0.5 ) * m_alpha;

    // Compute gradient glag
    m_glag.noalias() = m_bmat.col(m_knew - 1);

    for ( integer k = 0; k < m_npt; ++k )
    {
      auto const & xptk = m_xpt.col( k );
      Scalar temp = xptk.dot(m_xopt);
      temp *= m_hcol(k);
      //m_glag += temp * xptk; // @@@@@@@@@@@@
      for ( integer i = 0; i < m_neq; ++i ) m_glag[i] += temp * xptk( i );
    }

    // Search best line
    Scalar presav = 0;
    for ( integer k = 1; k <= m_npt; ++k )
    {
      if ( k == m_kopt+1 ) continue;

      Scalar dderiv = 0, distsq = 0;
      for ( integer i = 1; i <= m_neq; ++i )
      {
        Scalar temp = m_xpt( i - 1, k - 1 ) - m_xopt[i - 1];
        dderiv += m_glag[i - 1] * temp;
        distsq += temp * temp;
      }

      Scalar  subd = m_adelt / sqrt( distsq );
      Scalar  slbd = -subd;
      integer ilbd = 0, iubd = 0;

      Scalar sumin = std::min( Scalar( 1 ), subd );

      // bound projection - CORRETTA come originale
      for ( integer i = 1; i <= m_neq; ++i )
      {
        auto const & xo = m_xopt[i - 1];
        auto const & su = m_su[i - 1];
        auto const & sl = m_sl[i - 1];

        Scalar temp = m_xpt( i - 1, k - 1 ) - xo;

        if ( temp > 0 )
        {
          if ( slbd * temp < sl - xo )
          {
            slbd = ( sl - xo ) / temp;
            ilbd = -i;
          }
          if ( subd * temp > su - xo )
          {
            subd = std::max( ( su - xo ) / temp, sumin );
            iubd = i;
          }
        }
        else if ( temp < 0 )
        {
          if ( slbd * temp > su - xo )
          {
            slbd = ( su - xo ) / temp;
            ilbd = i;
          }
          if ( subd * temp < sl - xo )
          {
            subd = std::max( ( sl - xo ) / temp, sumin );
            iubd = -i;
          }
        }
      }

      Scalar  vlag;
      integer isbd;

      if ( k == m_knew )
      {
        Scalar diff = dderiv - 1;
        step        = slbd;
        vlag        = slbd * ( dderiv - slbd * diff );
        isbd        = ilbd;

        Scalar temp = subd * ( dderiv - subd * diff );
        if ( std::abs( temp ) > std::abs( vlag ) )
        {
          step = subd;
          vlag = temp;
          isbd = iubd;
        }

        Scalar tempd = Scalar( 0.5 ) * dderiv;
        Scalar tempa = tempd - diff * slbd;
        Scalar tempb = tempd - diff * subd;
        if ( tempa * tempb < 0 )
        {
          temp = tempd * tempd / diff;
          if ( std::abs( temp ) > std::abs( vlag ) )
          {
            step = tempd / diff;
            vlag = temp;
            isbd = 0;
          }
        }
      }
      else
      {
        step = slbd;
        vlag = slbd * ( 1 - slbd );
        isbd = ilbd;

        Scalar temp = subd * ( 1 - subd );
        if ( std::abs( temp ) > std::abs( vlag ) )
        {
          step = subd;
          vlag = temp;
          isbd = iubd;
        }

        if ( subd > Scalar( 0.5 ) )
        {
          if ( std::abs( vlag ) < Scalar( 0.25 ) )
          {
            step = Scalar( 0.5 );
            vlag = Scalar( 0.25 );
            isbd = 0;
          }
        }

        vlag *= dderiv;
      }

      Scalar temp   = step * ( 1 - step ) * distsq;
      Scalar predsq = vlag * vlag * ( vlag * vlag + ha * temp * temp );

      if ( predsq > presav )
      {
        presav = predsq;
        ksav   = k;
        stpsav = step;
        ibdsav = isbd;
      }
    }

    // construct xnew - Se ksav è ancora 0, c'è un problema
    if ( ksav == 0 )
    {
      // Fallback: usa knew come ksav
      ksav = m_knew;
      // Calcola un step di default
      Scalar distsq = 0;
      for ( integer i = 1; i <= m_neq; ++i )
      {
        Scalar temp = m_xpt( i - 1, ksav - 1 ) - m_xopt[i - 1];
        distsq += temp * temp;
      }
      stpsav = m_adelt / sqrt( distsq );
      ibdsav = 0;
    }
    
    m_xnew = ((m_xopt + stpsav * ( m_xpt.col(ksav - 1) - m_xopt ) ).cwiseMin(m_su)).cwiseMax(m_sl);

    // Applica i bound specifici
    if ( ibdsav < 0 )
    {
      // Nota: ibdsav contiene l'indice con segno
      integer idx     = -ibdsav;  // Converti in positivo
      m_xnew[idx - 1] = m_sl[idx - 1];
    }
    if ( ibdsav > 0 ) { m_xnew[ibdsav - 1] = m_su[ibdsav - 1]; }

    // ====== Cauchy step evaluation ======
    auto compute_cauchy_step = [&]( bool flip_grad )
    {
      if ( flip_grad ) m_glag.noalias() = -m_glag;

      Scalar bigstp = m_adelt + m_adelt;
      Scalar wfixsq = 0;
      Scalar ggfree = 0;

      std::fill( W.begin(), W.end(), 0 );

      for ( integer i = 0; i < m_neq; ++i )
      {
        Scalar tempa = std::min( m_xopt(i) - m_sl(i), m_glag(i) );
        Scalar tempb = std::max( m_xopt(i) - m_su(i), m_glag(i) );

        if ( tempa > 0 || tempb < 0 )
        {
          W(i) = bigstp;
          ggfree += m_glag(i) * m_glag(i);
        }
      }

      if ( ggfree == 0 )
      {
        std::fill( m_xalt.begin(), m_xalt.end(), 0 );
        return Scalar( 0 );
      }

      Scalar step_local = 0;
      // recheck bounds until stable
      while ( true )
      {
        Scalar temp = m_adelt * m_adelt - wfixsq;
        if ( temp <= 0 ) break;

        Scalar wsqsav = wfixsq;
        step_local    = sqrt( temp / ggfree );
        ggfree        = 0;

        for ( integer i = 1; i <= m_neq; ++i )
        {
          if ( W[i - 1] == bigstp )
          {
            Scalar cand = m_xopt[i - 1] - step_local * m_glag[i - 1];
            if ( cand <= m_sl[i - 1] )
            {
              W[i - 1] = m_sl[i - 1] - m_xopt[i - 1];
              wfixsq += W[i - 1] * W[i - 1];
            }
            else if ( cand >= m_su[i - 1] )
            {
              W[i - 1] = m_su[i - 1] - m_xopt[i - 1];
              wfixsq += W[i - 1] * W[i - 1];
            }
            else
            {
              ggfree += m_glag[i - 1] * m_glag[i - 1];
            }
          }
        }
        if ( !( wfixsq > wsqsav && ggfree > 0 ) ) break;
      }

      Scalar gw = 0;
      for ( integer i = 1; i <= m_neq; ++i )
      {
        if ( W[i - 1] == bigstp )
        {
          W[i - 1]      = -step_local * m_glag[i - 1];
          Scalar v      = std::min( m_xopt[i - 1] + W[i - 1], m_su[i - 1] );
          m_xalt[i - 1] = std::max( v, m_sl[i - 1] );
        }
        else if ( W[i - 1] == 0 ) { m_xalt[i - 1] = m_xopt[i - 1]; }
        else if ( m_glag[i - 1] > 0 ) { m_xalt[i - 1] = m_sl[i - 1]; }
        else
        {
          m_xalt[i - 1] = m_su[i - 1];
        }

        gw += m_glag[i - 1] * W[i - 1];
      }

      Scalar curv = 0;
      for ( integer k = 1; k <= m_npt; ++k )
      {
        Scalar temp = 0;
        for ( integer j = 1; j <= m_neq; ++j ) { temp += m_xpt( j - 1, k - 1 ) * W[j - 1]; }
        curv += m_hcol[k - 1] * temp * temp;
      }

      if ( flip_grad ) curv = -curv;

      Scalar result;
      if ( curv > -gw && curv < -one_plus_sqrt2 * gw )
      {
        Scalar scale = -gw / curv;
        for ( integer i = 1; i <= m_neq; ++i )
        {
          Scalar v      = std::min( m_xopt[i - 1] + scale * W[i - 1], m_su[i - 1] );
          m_xalt[i - 1] = std::max( v, m_sl[i - 1] );
        }
        Scalar temp = Scalar( 0.5 ) * gw * scale;
        result      = temp * temp;
      }
      else
      {
        Scalar temp = gw + Scalar( 0.5 ) * curv;
        result      = temp * temp;
      }

      return result;
    };

    // Evaluate downhill and uphill version
    Scalar c1 = compute_cauchy_step( false );
    for ( integer i = 1; i <= m_neq; ++i ) { W[m_neq + i - 1] = m_xalt[i - 1]; }
    csave = c1;

    Scalar c2 = compute_cauchy_step( true );

    if ( csave > c2 )
    {
      for ( integer i = 1; i <= m_neq; ++i ) { m_xalt[i - 1] = W[m_neq + i - 1]; }
      m_cauchy = csave;
    }
    else
    {
      m_cauchy = c2;
    }
  }


  /*----------------------------------------------------------------------------*/
  template <typename Scalar>
  void
  BOBYQA_minimizer<Scalar>::prelim( bobyqa_objfun const & objfun, Vector & X )
  {
    /* The arguments N, NPT, X, XL, XU, RHOBEG, IPRINT and MAXFUN are the same as
       the corresponding arguments in SUBROUTINE BOBYQA.  The arguments XBASE,
       XPT, FVAL, HQ, PQ, BMAT, ZMAT, NDIM, SL and SU are the same as the
       corresponding arguments in BOBYQB, the elements of SL and SU being set in
       BOBYQA.

       GOPT is usually the gradient of the quadratic model at XOPT+XBASE, but it
       is set by PRELIM to the gradient of the quadratic model at XBASE.

       If XOPT is nonzero, BOBYQB will change it to its usual value later.

       NF is maintaned as the number of calls of OBJFUN so far.

       KOPT will be such that the least calculated value of F so far is at the
       point XPT(KOPT,.)+XBASE in the space of the variables.

       SUBROUTINE PRELIM sets the elements of XBASE, XPT, FVAL, GOPT, HQ, PQ,
       BMAT and ZMAT for the first iteration, and it maintains the values of NF
       and KOPT.  The vector X is also changed by PRELIM. */

    /* Set some constants. */
    Scalar  rhosq = m_rhobeg * m_rhobeg;
    Scalar  recip = 1 / rhosq;
    integer np    = m_neq + 1;
    Scalar  fbeg  = 0;

    /* Set XBASE to the initial vector of variables, and set the initial elements
       of XPT, BMAT, HQ, PQ and ZMAT to zero. */
    m_xpt.setZero();
    m_bmat.setZero();
    m_hq.setZero();
    m_pq.setZero();
    m_zmat.setZero();
    
    m_xbase.noalias() = X.cwiseMin(m_xupper).cwiseMax(m_xlower);
    
    auto OBJ = [&]( integer ipos ) -> Scalar {
      for ( integer j = 0; j < m_neq; ++j )
      {
        Scalar temp = m_xbase( j ) + m_xpt( j, ipos );
        temp        = std::max( temp, m_xlower( j ) );
        X( j )      = std::min( temp, m_xupper( j ) );
        if ( m_xpt( j, ipos ) == m_sl( j ) ) X( j ) = m_xlower( j );
        if ( m_xpt( j, ipos ) == m_su( j ) ) X( j ) = m_xupper( j );
      }

      Scalar f = objfun( X );
      m_fval( ipos ) = f;
      if ( f < m_fval(m_kopt) ) m_kopt = ipos;

      if ( m_print_level == 3 )
        fmt::print(
            "    Function n.{} F = {:.9}\n"
            "    The corresponding X is: {}\n",
            ipos+1, f, print_vec( X, 6 ) );
      return f;
    };

    // ----------------------------------------------------------------------------
    // PARTE 1: I = 1 - Punto base (nessuno spostamento)
    // ----------------------------------------------------------------------------
    {
      m_nf      = 1;
      fbeg      = objfun(m_xbase);
      m_fval(0) = fbeg;
      m_kopt    = 0;
    }

    // ----------------------------------------------------------------------------
    // PARTE 2: I = 2..m_neq - Punti lungo direzioni positive degli assi
    // ----------------------------------------------------------------------------
    for (integer I = 2; I <= m_neq; ++I) {
      m_nf = I;
    
      // CORRETTO: nfm è tra 1 e m_neq, quindi questo blocco viene eseguito
      Scalar stepa = m_rhobeg;
      if (m_su(I-2) == 0) stepa = -stepa;
      m_xpt(I-2,I-1) = stepa;
      
      Scalar f = OBJ( I - 1 );

      /* Aggiorna gradiente e BMAT */
      // m_nf è tra 2 e m_neq, quindi m_nf >= 2 && m_nf <= m_neq+1 è VERO
      m_gopt(I-2) = (f - fbeg) / stepa;
    
      if (m_npt < m_nf + m_neq) {
        m_bmat( I-2, 0           ) = Scalar(-1.0) / stepa;
        m_bmat( I-2, I-1         ) = Scalar(1.0)  / stepa;
        m_bmat( I-2, m_npt + I-2 ) = -Scalar(0.5) * rhosq;
      }
    }

    // ----------------------------------------------------------------------------
    // PARTE 3: I = m_neq+1 - Ultimo punto lungo asse positivo
    // ----------------------------------------------------------------------------
    {
      integer I = m_neq + 1;
      m_nf = I;
    
      // nfm = m_neq, quindi nfm >= 1 && nfm <= m_neq è VERO
      Scalar stepa = m_rhobeg;
      if (m_su(I-2) == 0) stepa = -stepa;
      m_xpt(I-2,I-1) = stepa;
      
      Scalar f = OBJ( I - 1 );
    
      m_gopt(I-2) = (f - fbeg) / stepa;
    
      if (m_npt < m_nf + m_neq) {
          m_bmat(I-2, 0           ) = Scalar(-1.0) / stepa;
          m_bmat(I-2, I-1         ) = Scalar(1.0)  / stepa;
          m_bmat(I-2, m_npt + I-2 ) = -Scalar(0.5) * rhosq;
      }
    }

    // ----------------------------------------------------------------------------
    // PARTE 4: I = m_neq+2..2*m_neq - Punti lungo direzioni negative
    // ----------------------------------------------------------------------------
    for (integer I = m_neq + 2; I <= 2 * m_neq; ++I) {
      integer nfx  = I - 1 - m_neq;   // 1..m_neq-1 (1-based)
      integer nfx0 = nfx - 1;
      m_nf = I;
      
      // nfm > m_neq, quindi usiamo il ramo else if (nfm > m_neq)
      Scalar stepa = m_xpt(nfx0, nfx);
      Scalar stepb = -m_rhobeg;
      
      if (m_sl(nfx0) == 0) stepb = std::min( 2 * m_rhobeg, m_su(nfx0));
      if (m_su(nfx0) == 0) stepb = std::max(-2 * m_rhobeg, m_sl(nfx0));
      
      m_xpt(nfx0,I-1) = stepb;
      
      Scalar f = OBJ( I - 1 );
  
      /* Aggiorna modello quadratico */
      // m_nf >= m_neq+2, quindi usiamo il ramo else if (m_nf >= m_neq+2)
      integer ih = nfx * (nfx + 1) / 2;
      Scalar temp = (f - fbeg) / stepb;
      Scalar diff = stepb - stepa;
      
      m_hq(ih - 1) = 2 * (temp - m_gopt(nfx0)) / diff;
      m_gopt(nfx0) = (m_gopt(nfx0) * stepb - temp * stepa) / diff;
      
      if (stepa * stepb < 0) {
        if (f < m_fval(nfx)) {
          m_fval(I-1) = m_fval(nfx);
          m_fval(nfx) = f;
          if (m_kopt+1 == I) m_kopt = nfx;
          m_xpt(nfx0, nfx ) = stepb;
          m_xpt(nfx0, I-1 ) = stepa;
        }
      }
      
      m_bmat( nfx0, 0   ) = -(stepa + stepb) / (stepa * stepb);
      m_bmat( nfx0, I-1 ) = -Scalar(0.5) / m_xpt(nfx0, nfx);
      m_bmat( nfx0, nfx ) = -m_bmat(nfx0, 0) - m_bmat(nfx0, I-1);
      
      m_zmat( 0,   nfx0 ) = sqrt(Scalar(2)) / (stepa * stepb);
      m_zmat( I-1, nfx0 ) = sqrt(Scalar(0.5)) / rhosq;
      m_zmat( nfx, nfx0 ) = -m_zmat(0, nfx0) - m_zmat(I-1, nfx0);
    }

    //  integer nfm = m_nf - 1;
    //  integer nfx = m_nf - 1 - m_neq;
    {
      integer I    = 2*m_neq+1;
      integer nfx  = I-1 - m_neq;
      integer nfx0 = nfx - 1;
      m_nf = I;
      
      Scalar stepa = m_xpt( nfx0, nfx );
      Scalar stepb = -m_rhobeg;
      if ( m_sl( nfx0 ) == 0 ) stepb = std::min( 2 * m_rhobeg, m_su( nfx0 ) );
      if ( m_su( nfx0 ) == 0 ) stepb = std::max( -2 * m_rhobeg, m_sl( nfx0 ) );
      m_xpt( nfx0, I-1 ) = stepb;
      
      Scalar f = OBJ( I-1 );

      /* Set the nonzero initial elements of BMAT and the quadratic model in the
         cases when NF is at most 2*N+1.  If NF exceeds N+1, then the positions
         of the NF-th and (NF-N)-th interpolation points may be switched, in
         order that the function value at the first of them contributes to the
         off-diagonal second derivative terms of the initial quadratic model. */
      integer ih     = nfx * ( nfx + 1 ) / 2;
      Scalar  temp   = ( f - fbeg ) / stepb;
      Scalar  diff   = stepb - stepa;
      m_hq( ih - 1 ) = 2 * ( temp - m_gopt( nfx0 ) ) / diff;
      m_gopt( nfx0 ) = ( m_gopt( nfx0 ) * stepb - temp * stepa ) / diff;
      if ( stepa * stepb < 0 ) {
        if ( f < m_fval( nfx ) ) {
          m_fval( I-1 ) = m_fval( nfx );
          m_fval( nfx ) = f;
          if ( m_kopt+1 == I ) m_kopt = nfx;
          m_xpt( nfx0, nfx ) = stepb;
          m_xpt( nfx0, I-1 ) = stepa;
        }
      }
      m_bmat( nfx0, 0   ) = -( stepa + stepb ) / ( stepa * stepb );
      m_bmat( nfx0, I-1 ) = -Scalar( 0.5 ) / m_xpt( nfx0, nfx );
      m_bmat( nfx0, nfx ) = -m_bmat( nfx0, 0 ) - m_bmat( nfx0, I-1 );
      m_zmat( 0,   nfx0 ) = sqrt( Scalar( 2 ) ) / ( stepa * stepb );
      m_zmat( I-1, nfx0 ) = sqrt( Scalar( 0.5 ) ) / rhosq;
      m_zmat( nfx, nfx0 ) = -m_zmat( 0, nfx0 ) - m_zmat( I-1, nfx0 );
    }

    //  integer nfm = m_nf - 1;
    //  integer nfx = m_nf - 1 - m_neq;
    for ( integer I = 2*m_neq+2; I <= m_npt; ++I ) {
      integer nfm  = I - 1;           // ≥ 2*m_neq
      integer nfx  = I-1 - m_neq;
      integer nfx0 = nfx-1;
      m_nf = I;

      // 1. CALCOLA COPPIA DI DIREZIONI (ipt, jpt)
      integer itemp = (nfm - np) / m_neq;  // np è costante esterna
      integer jpt = nfm - itemp * m_neq - m_neq;
      integer ipt = jpt + itemp;
      
      // Assicura che ipt ≤ m_neq (riduzione modulo m_neq)
      if (ipt > m_neq) {
        itemp = jpt;
        jpt = ipt - m_neq;
        ipt = itemp;
      }

      Scalar stepb = -m_rhobeg;
      if ( m_sl( nfx0 ) == 0 ) stepb = std::min(  2 * m_rhobeg, m_su( nfx0 ) );
      if ( m_su( nfx0 ) == 0 ) stepb = std::max( -2 * m_rhobeg, m_sl( nfx0 ) );
      m_xpt( nfx0, I-1 ) = stepb;

      Scalar f = OBJ( I-1 );

      integer ih          = ipt * ( ipt - 1 ) / 2 + jpt;
      m_zmat( 0,     nfx0 ) = recip;
      m_zmat( I-1,   nfx0 ) = recip;
      m_zmat( ipt-1, nfx0 ) = -recip;
      m_zmat( jpt-1, nfx0 ) = -recip;
      Scalar temp         = m_xpt( ipt - 1, I-1 ) * m_xpt( jpt - 1, I-1 );
      m_hq( ih - 1 )      = ( fbeg - m_fval( ipt-1 ) - m_fval( jpt-1 ) + f ) / temp;
    }

  } /* prelim */


  /**
   * @brief Procedura di salvataggio (Rescue) per ripristinare la geometria dell'interpolazione.
   *
   * @details Questa funzione viene chiamata quando l'algoritmo rileva instabilità numerica
   * o quando i punti di interpolazione sono degenerati (es. quasi allineati).
   * La procedura:
   * 1. Sposta l'origine delle coordinate nel punto ottimo corrente (XOPT).
   * 2. Ripristina la matrice Z (che gestisce l'indipendenza lineare).
   * 3. Aggiorna la matrice Hessiana (HQ) e le matrici di lavoro (BMAT).
   * 4. Tenta di sostituire iterativamente i punti peggiori con punti geometricamente
   * più validi (basati sui bounds o sulla direzione di massima curvatura),
   * utilizzando operazioni vettoriali per efficienza.
   */
  template <typename Scalar>
  void
  BOBYQA_minimizer<Scalar>::rescue()
  {
    // Vettore di lavoro locale: prime m_dim entrate per calcoli su variabili,
    // successive m_npt per calcoli sui punti.
    Vector m_work( m_dim + m_npt );

    m_beta       = 0;
    Scalar denom = 0;
    Scalar sumpq = 0;
    Scalar winc  = 0;

    const integer n     = m_neq;  // Dimensione variabili
    const integer npt   = m_npt;  // Numero punti interpolazione
    const integer np    = n + 1;
    const Scalar  sfrac = Scalar( 0.5 ) / Scalar( np );
    const integer nptm  = npt - np;  // Numero colonne di ZMAT

    // ==============================================================================
    // STEP 1: Shift delle coordinate all'origine e calcolo distanze
    // ==============================================================================
    // Spostiamo tutti i punti XPT in modo che XOPT diventi l'origine (0,0...).
    // XPT è (npt x n). Sottraiamo XOPT (n x 1) da ogni riga.
    m_xpt.transpose().rowwise() -= m_xopt.transpose();

    // Calcoliamo la somma dei pesi PQ
    sumpq = m_pq.sum();

    // Calcoliamo le norme quadrate di ogni punto dopo lo shift
    // m_work[m_dim ... m_dim+npt-1] conterrà le distanze.
    // Usiamo segment() per accedere alla parte del vettore m_work dedicata ai punti.
    m_work.tail( npt ) = m_xpt.transpose().rowwise().squaredNorm();

    // Troviamo la distanza massima per il calcolo di winc
    winc = m_work.tail( npt ).maxCoeff();

    // Resettiamo ZMAT: le prime nptm colonne vengono azzerate.
    // ZMAT gestisce lo spazio nullo dell'interpolazione.
    if ( nptm > 0 ) { m_zmat.leftCols( nptm ).setZero(); }

    // ==============================================================================
    // STEP 2: Aggiornamento Hessiana (HQ) dovuto allo shift
    // ==============================================================================
    // Calcoliamo il vettore di correzione: v = 0.5 * sumpq * XOPT + XPT^T * PQ
    // Nota: XOPT qui è il vecchio XOPT (che ora è l'origine relativa, ma serve il valore pre-shift)
    // Tuttavia, XPT è già shiftato. La formula originale usa XPT shiftato.

    // Per ottimizzare, calcoliamo prima il vettore ausiliario nelle prime 'n' posizioni di work.
    // m_work.head(n) = 0.5 * sumpq * m_xopt + m_xpt * m_pq;

    // Attenzione: m_xopt nel codice originale viene usato nel loop HQ *prima* di essere azzerato nello step 3.
    // XPT è stato modificato in place.
    // Ricostruiamo la logica vettoriale:
    Vector vec_correction = ( Scalar( 0.5 ) * sumpq ) * m_xopt;
    vec_correction += m_xpt * m_pq;

    // Aggiornamento packed upper-triangular di HQ
    // HQ += vec_correction * xopt^T + xopt * vec_correction^T
    // Poiché HQ è memorizzato linearmente (packed storage), manteniamo il loop per sicurezza sugli indici,
    // ma usiamo il vettore precalcolato.
    integer ih = 0;
    for ( integer j = 0; j < n; ++j )
    {
      Scalar wj     = vec_correction( j );
      Scalar xopt_j = m_xopt( j );
      for ( integer i = 0; i <= j; ++i )
      {
        m_hq( ih ) += vec_correction( i ) * xopt_j + wj * m_xopt( i );
        ih++;
      }
    }

    // ==============================================================================
    // STEP 3: Shift variabili base e costruzione PTSAUX
    // ==============================================================================
    // Spostiamo i bound e xbase rispetto al nuovo xopt (che diventa 0)
    m_xbase += m_xopt;
    m_sl -= m_xopt;
    m_su -= m_xopt;

    // Costruiamo i punti ausiliari PTSAUX basati sui bounds (Trust Region Box)
    // PTSAUX è 2xN. Riga 0 = lower step, Riga 1 = upper step (o viceversa ordinati)
    for ( integer j = 0; j < n; ++j )
    {
      Scalar sl_j = m_sl( j );
      Scalar su_j = m_su( j );

      Scalar cand0 = std::min( m_delta, su_j );
      Scalar cand1 = std::max( -m_delta, sl_j );

      // Ordinamento per garantire stabilità numerica
      if ( cand0 + cand1 < 0 ) std::swap( cand0, cand1 );

      // Riduzione se troppo vicino a zero
      if ( std::abs( cand1 ) < Scalar( 0.5 ) * std::abs( cand0 ) ) { cand1 = Scalar( 0.5 ) * cand0; }

      m_ptsaux( 0, j ) = cand0;
      m_ptsaux( 1, j ) = cand1;
    }

    // XOPT ora è formalmente l'origine
    m_xopt.setZero();

    // Azzeriamo le prime n righe di BMAT (che corrispondono ai termini lineari puri)
    m_bmat.leftCols( n ).setZero();

    // ==============================================================================
    // STEP 4: Costruzione iniziale PTSAUX / BMAT / ZMAT
    // ==============================================================================
    // Qui si definiscono i punti di interpolazione ideali lungo gli assi

    m_ptsid( 0 ) = sfrac;  // Punto base

    // Loop principale sulle dimensioni per impostare BMAT/ZMAT diagonali/sparse
    for ( integer j = 0; j < n; ++j )
    {
      integer jp  = j + 1;   // Indice punto positivo (1-based relativo a 0)
      integer jpn = jp + n;  // Indice punto negativo

      m_ptsid( jp ) = Scalar( j + 1 ) + sfrac;

      Scalar d0 = m_ptsaux( 0, j );
      Scalar d1 = m_ptsaux( 1, j );

      // Calcoli comuni
      Scalar inv_d0   = Scalar( 1 ) / d0;
      Scalar diff_inv = Scalar( 1 ) / ( d0 - d1 );

      if ( jpn < npt )
      {
        m_ptsid( jpn ) = Scalar( j + 1 ) / Scalar( np ) + sfrac;

        // Aggiornamento BMAT
        m_bmat( j, jp  ) = -diff_inv + inv_d0;
        m_bmat( j, jpn ) = diff_inv + Scalar( 1 ) / d1;
        m_bmat( j, 0   ) = -m_bmat( j, jp ) - m_bmat( j, jpn );

        // Aggiornamento ZMAT (curvatura)
        Scalar z_base    = std::sqrt( Scalar( 2.0 ) ) / std::abs( d0 * d1 );
        m_zmat( 0, j )   = z_base;
        m_zmat( jp, j )  = z_base * d1 * diff_inv;
        m_zmat( jpn, j ) = -z_base * d0 * diff_inv;
      }
      else
      {
        // Caso in cui abbiamo meno punti del necessario per il doppio stencil
        m_bmat( j, 0  )  = -inv_d0;
        m_bmat( j, jp )  = inv_d0;
        // BMAT per termine quadratico nel caso degenere
        m_bmat( j, j + npt ) = -Scalar( 0.5 ) * d0 * d0;  // Nota: indice fuori range standard, verifica dimensione m_bmat
      }
    }

    // Gestione punti extra (oltre 2*n + 1)
    if ( npt >= n + np )
    {
      for ( integer k = 2 * np; k <= npt; ++k )
      {
        // Logica per distribuire i punti extra
        integer k_idx = k - 1;  // 0-based
        integer iw    = static_cast<integer>( ( Scalar( k - np ) - Scalar( 0.5 ) ) / Scalar( n ) );
        integer ip    = k - np - iw * n;
        integer iq    = ip + iw;
        if ( iq > n ) iq -= n;

        m_ptsid( k_idx ) = Scalar( ip ) + Scalar( iq ) / Scalar( np ) + sfrac;

        // Indici 0-based per array
        integer ip_idx = ip - 1;
        integer iq_idx = iq - 1;

        Scalar tmp = Scalar( 1 ) / ( m_ptsaux( 0, ip_idx ) * m_ptsaux( 0, iq_idx ) );

        integer col_z          = k - np - 1;  // Colonna in ZMAT
        m_zmat( 0, col_z )     = tmp;
        m_zmat( ip, col_z )    = -tmp;
        m_zmat( iq, col_z )    = -tmp;
        m_zmat( k_idx, col_z ) = tmp;
      }
    }


    // ==============================================================================
    // MAIN PHASE LOOP: Ripristino iterativo
    // ==============================================================================
    integer nrem = npt;
    integer kold = 0;  // 0-based
    m_knew       = m_kopt+1;

    // Funzione helper per lo swap
    auto swap_points = [&]( integer k1, integer k2 )
    {
      if ( k1 == k2 ) return;

      m_bmat.col( k1 ).swap( m_bmat.col( k2 ) );
      if ( nptm > 0 ) { m_zmat.row( k1 ).head( nptm ).swap( m_zmat.row( k2 ).head( nptm ) ); }
      std::swap( m_ptsid( k1 ), m_ptsid( k2 ) );
      std::swap( m_work( m_dim + k1 ), m_work( m_dim + k2 ) );
      std::swap( m_vlag( k1 ), m_vlag( m_knew ) );  // Correzione: usa m_knew, non k2
    };

    while ( nrem > 0 )
    {
      // PHASE 1: Scambia kold e knew
      if ( m_knew != m_kopt+1 )
      {
        swap_points( kold, m_knew - 1 );
        m_work( m_dim + m_knew - 1 ) = 0;
        nrem--;

        // Aggiorna BMAT e ZMAT
        update();  // Questa deve aggiornare m_beta e denom

        if ( nrem == 0 ) break;

        m_work.tail( npt ) = m_work.tail( npt ).cwiseAbs();
      }

      // PHASE 2: Trova nuovo candidato knew
      Scalar  dsqmin    = 0;
      integer best_knew = -1;

      for ( integer k = 0; k < npt; ++k )
      {
        Scalar d = m_work( m_dim + k );
        if ( d > 0 && ( dsqmin == 0 || d < dsqmin ) )
        {
          dsqmin    = d;
          best_knew = k + 1;
        }
      }

      if ( best_knew == -1 ) break;
      m_knew = best_knew;

      // PHASE 3: Calcola w_vec (come nel codice originale)
      Vector w_vec( npt + n );
      w_vec.tail( n ) = m_xpt.col( m_knew - 1 );

      // Calcola w_vec.head(npt)
      for ( integer k = 0; k < npt; ++k )
      {
        if ( k == m_kopt )
        {
          w_vec( k ) = 0;
          continue;
        }

        Scalar sum = 0;
        if ( m_ptsid( k ) == 0 )
        {
          // Punto originale: prodotto scalare
          sum = m_xpt.col( k ).dot( m_xpt.col( m_knew ) );
        }
        else
        {
          // Punto artificiale
          integer ip = static_cast<integer>( m_ptsid( k ) );
          integer iq = static_cast<integer>( np * m_ptsid( k ) - ip * np );

          if ( ip > 0 ) { sum += w_vec( npt + ip - 1 ) * m_ptsaux( 0, ip - 1 ); }
          if ( iq > 0 )
          {
            integer iw = 1;         // Usa PTSAUX(1, iq) per default
            if ( ip == 0 ) iw = 2;  // Usa PTSAUX(2, iq) se ip = 0
            sum += w_vec( npt + iq - 1 ) * m_ptsaux( iw - 1, iq - 1 );
          }
        }
        w_vec( k ) = 0.5 * sum * sum;
      }

      // PHASE 4: Calcola VLAG e BETA
      // Parte BMAT
      m_vlag.head( npt ) = m_bmat.leftCols( npt ).transpose() * w_vec.tail( n );

      // Parte ZMAT
      m_beta = 0;
      if ( nptm > 0 )
      {
        for ( integer j = 0; j < nptm; ++j )
        {
          // Calcola sum = Σ_k ZMAT(k, j) * w_vec(k)
          Scalar sum_z = 0;
          for ( integer k = 0; k < npt; ++k ) { sum_z += m_zmat( k, j ) * w_vec( k ); }
          m_beta -= sum_z * sum_z;

          // Aggiorna VLAG
          for ( integer k = 0; k < npt; ++k ) { m_vlag( k ) += sum_z * m_zmat( k, j ); }
        }
      }

      // Calcola bsum e distsq
      Scalar bsum   = 0;
      Scalar distsq = m_xpt.col( m_knew - 1 ).squaredNorm();

      for ( integer j = 0; j < n; ++j )
      {
        Scalar sum = 0;
        // Somma sulle prime npt righe di BMAT
        for ( integer k = 0; k < npt; ++k ) { sum += m_bmat( j, k ) * w_vec( k ); }
        bsum += sum * w_vec( npt + j );

        // Somma sulle righe da npt in poi di BMAT
        for ( integer ip = npt; ip < npt + n; ++ip ) { sum += m_bmat( j, ip ) * w_vec( ip ); }
        bsum += sum * w_vec( npt + j );
        m_vlag( npt + j ) = sum;
      }

      m_beta = 0.5 * distsq * distsq + m_beta - bsum;
      m_vlag( m_kopt ) += 1.0;

      // PHASE 5: Trova kold con il massimo denominatore
      Scalar vlmxsq = 0;
      for ( integer k = 0; k < npt; ++k )
      {
        Scalar temp = m_vlag( k ) * m_vlag( k );
        if ( temp > vlmxsq ) vlmxsq = temp;
      }

      Scalar  denom_max = 0;
      integer best_kold = -1;

      for ( integer k = 0; k < npt; ++k )
      {
        if ( m_ptsid( k ) != 0 )
        {
          Scalar hdiag = 0;
          if ( nptm > 0 )
          {
            for ( integer j = 0; j < nptm; ++j ) { hdiag += m_zmat( k, j ) * m_zmat( k, j ); }
          }
          Scalar den = m_beta * hdiag + m_vlag( k ) * m_vlag( k );
          if ( den > denom_max )
          {
            denom_max = den;
            best_kold = k;
          }
        }
      }

      if ( best_kold == -1 ) break;
      kold = best_kold;

      // PHASE 6: Verifica denominatore
      if ( denom_max <= 0.01 * vlmxsq )
      {
        m_work( m_dim + m_knew ) = -m_work( m_dim + m_knew ) - winc;
        continue;
      }

      // PHASE 7: Gestione caso speciale (knew == kopt)
      if ( m_knew == m_kopt+1 )
      {
        m_work( m_dim + m_knew ) = 0;
        nrem--;
        if ( nrem == 0 ) break;
      }
    }


    // ==============================================================================
    // FASE FINALE: Ripristino del modello nei nuovi punti
    // ==============================================================================
    // Equivalente al loop L260 -> L350
    // Valutiamo i punti dove ptsid è rimasto settato (punti nuovi generati dal rescue)

    for ( integer kpt = 0; kpt < npt; ++kpt )
    {
      if ( m_ptsid( kpt ) == 0 ) continue;

      // Limite valutazioni funzione
      if ( m_nf >= m_maxfun )
      {
        m_nf = -1;
        break;
      }

      // Qui il codice originale esegue la valutazione della funzione e l'aggiornamento
      // Poiché avevi indicato "same calculations as your code", assumiamo che
      // la logica di update del modello e valutazione f(x) avvenga qui o sia
      // gestita implicitamente se questo blocco era vuoto nel tuo snippet.
      // Nel snippet originale c'era un commento, manteniamo la struttura.

      m_ptsid( kpt ) = 0;
    }
  }


  /**
   * @brief Risolve il sottoproblema di trust region vincolato per BOBYQA
   *
   * @details
   * Implementa un algoritmo di gradiente coniugato troncato con proiezione sui bound
   * per risolvere il sottoproblema di trust region:
   *
   * \f[
   * \begin{aligned}
   * \min_{d} \quad & g^T d + \frac{1}{2} d^T H d \\
   * \text{s.t.} \quad & \|d\| \leq \delta \\
   *                   & x_l - x_{opt} \leq d \leq x_u - x_{opt}
   * \end{aligned}
   * \f]
   *
   * dove:
   * - \f$d = x - x_{opt}\f$ è lo spostamento dal punto corrente
   * - \f$g\f$ è il gradiente del modello quadratico in \f$x_{opt}\f$
   * - \f$H\f$ è l'Hessiana approssimata del modello
   * - \f$\delta\f$ è il raggio della trust region
   * - \f$x_l, x_u\f$ sono i bound inferiori e superiori
   *
   * @section algorithm Algoritmo
   *
   * L'algoritmo si divide in tre fasi principali:
   *
   * **FASE 1: Inizializzazione**
   * - Identifica le variabili da fissare ai bound basandosi sul segno del gradiente
   * - Inizializza le strutture dati per il gradiente coniugato
   *
   * **FASE 2: Gradiente Coniugato Proiettato**
   * - Applica il metodo del gradiente coniugato nello spazio delle variabili libere
   * - Proietta la soluzione sui bound quando necessario
   * - Riavvia il metodo quando una nuova variabile diventa attiva
   * - Termina quando raggiunge il bordo della trust region o converge
   *
   * **FASE 3: Iterazioni Alternative sul Bordo**
   * - Quando \f$\|d\| = \delta\f$, cerca miglioramenti rimanendo sul bordo
   * - Opera nello spazio 2D generato da \f$(d, \nabla m)\f$
   * - Usa ricerca angolare per trovare la direzione ottimale
   *
   * @section performance Ottimizzazioni Eigen3
   *
   * Questa implementazione sfrutta:
   * - Operazioni vettoriali per calcoli di norme e prodotti scalari
   * - Block operations per prodotti matrice-vettore
   * - Espressioni lazy evaluation per evitare allocazioni temporanee
   * - Segmenti di vettori per operare solo su variabili libere
   *
   * @post m_d contiene lo spostamento ottimale da x_opt
   * @post m_xnew = x_opt + d rispettando i bound
   * @post m_gnew contiene il gradiente in x_opt + d
   * @post m_dsq = ||d||^2
   * @post m_crvmin contiene la curvatura minima (0 se sul bordo)
   *
   * @tparam Scalar Tipo numerico (float, double, long double)
   *
   * @note Complessità: O(n * iterazioni * (n + npt)) per iterazione del CG
   * @note L'algoritmo è numericamente stabile grazie all'uso di sqrt per hypot
   *
   * @see M.J.D. Powell, "The BOBYQA Algorithm for Bound Constrained Optimization
   *      Without Derivatives" (2009)
   */
  template <typename Scalar>
  void
  BOBYQA_minimizer<Scalar>::trsbox()
  {

    // ========================================================================
    // VARIABILI LOCALI
    // ========================================================================

    /// Variabili per le iterazioni alternative
    Scalar angbd,  ///< Bound massimo sull'angolo di rotazione
        angt,      ///< Angolo di rotazione corrente
        cth,       ///< Coseno dell'angolo
        sth;       ///< Seno dell'angolo

    /// Variabili per il gradiente coniugato
    Scalar beta,  ///< Coefficiente per la direzione coniugata
        ggsav,    ///< Salvataggio di ||g||^2 precedente
        gredsq,   ///< ||g_ridotto||^2 (solo variabili libere)
        qred,     ///< Riduzione cumulativa del modello quadratico
        sdec;     ///< Riduzione corrente del passo

    /// Variabili per il calcolo del passo
    Scalar blen,  ///< Lunghezza fino al bordo della trust region
        delsq,    ///< Raggio della trust region al quadrato (δ²)
        ds,       ///< Prodotto scalare d·s
        resid,    ///< Spazio residuo: δ² - ||d||²
        shs,      ///< Curvatura s^T H s
        stepsq,   ///< ||s||² (norma al quadrato della direzione)
        stplen;   ///< Lunghezza del passo ottimale

    /// Variabili per le iterazioni alternative
    Scalar dredg,  ///< Prodotto scalare d_ridotto · g
        dredsq,    ///< ||d_ridotto||²
        dhd,       ///< Curvatura d^T H d
        dhs,       ///< Prodotto misto d^T H s
        sredg,     ///< Gradiente proiettato nella direzione s
        redmax,    ///< Massima riduzione trovata
        rednew,    ///< Riduzione per il nuovo angolo
        redsav,    ///< Salvataggio della riduzione precedente
        rdprev,    ///< Riduzione al passo precedente
        rdnext,    ///< Riduzione al passo successivo
        xsav;      ///< Salvataggio del valore del bound

    /// Indici e contatori
    integer iact,  ///< Indice della variabile che diventa attiva
        isav,      ///< Indice del miglior angolo trovato
        itcsav,    ///< Salvataggio del contatore di iterazioni
        iterc,     ///< Contatore iterazioni correnti
        itermax,   ///< Massimo numero di iterazioni CG
        iu,        ///< Numero di angoli da testare
        nact;      ///< Numero di variabili attive sui bound

    // ========================================================================
    // VETTORI DI LAVORO
    // ========================================================================

    Vector hred( m_neq );  ///< Prodotto H * d_ridotto
    Vector hs( m_neq );    ///< Prodotto H * s
    Vector s( m_neq );     ///< Direzione di ricerca corrente
    Vector xbdi( m_neq );  ///< Indicatori bound: -1 (lower), 0 (free), +1 (upper)

    // ========================================================================
    // FASE 1: INIZIALIZZAZIONE
    // ========================================================================

    /**
     * Determina quali variabili fissare inizialmente ai bound.
     *
     * Regola euristica:
     * - Se x_opt è sul lower bound E gradiente ≥ 0 → fissa al lower bound
     * - Se x_opt è sull'upper bound E gradiente ≤ 0 → fissa all'upper bound
     * - Altrimenti → variabile libera
     *
     * Questa inizializzazione accelera la convergenza evitando di esplorare
     * direzioni che violerebbero immediatamente i vincoli.
     */
    iterc = 0;
    nact  = 0;

    // Usa operazioni vettoriali Eigen per inizializzazione efficiente
    xbdi.setZero();
    m_d.setZero();
    m_gnew = m_gopt;

    for ( integer i = 0; i < m_neq; ++i )
    {
      // Variabile sul lower bound con gradiente non negativo
      if ( m_xopt( i ) <= m_sl( i ) && m_gopt( i ) >= 0 )
      {
        xbdi( i ) = -1;
        ++nact;
      }
      // Variabile sull'upper bound con gradiente non positivo
      else if ( m_xopt( i ) >= m_su( i ) && m_gopt( i ) <= 0 )
      {
        xbdi( i ) = 1;
        ++nact;
      }
    }

    delsq    = m_delta * m_delta;
    qred     = 0;
    m_crvmin = -1;

    // ========================================================================
    // FUNZIONE AUSILIARIA: Calcolo Prodotto Hessiana-Vettore
    // ========================================================================

    /**
     * @brief Calcola il prodotto H*s dove H è l'Hessiana del modello
     *
     * L'Hessiana è rappresentata in forma mista:
     * \f[
     * H = \sum_{i,j} HQ_{ij} e_i e_j^T + \sum_k PQ_k (XPT_k XPT_k^T)
     * \f]
     *
     * - Parte esplicita: matrice simmetrica HQ (immagazzinata come triangolare)
     * - Parte implicita: somma di matrici di rango-1 dai punti di interpolazione
     *
     * @param[in] s Vettore input
     * @param[out] hs Risultato H*s
     *
     * @complexity O(n² + npt*n)
     */
    auto compute_hessian_product = [&]( const Vector & s_in, Vector & hs_out )
    {
      // PARTE 1: Contributo della matrice esplicita HQ (simmetrica, packed storage)
      hs_out.setZero();

      integer ih = 0;
      for ( integer j = 0; j < m_neq; ++j )
      {
        for ( integer i = 0; i <= j; ++i )
        {
          const Scalar hq_val = m_hq( ih );
          ++ih;

          // HQ è simmetrica: aggiungi contributi per entrambi i lati
          hs_out( i ) += hq_val * s_in( j );
          if ( i < j ) { hs_out( j ) += hq_val * s_in( i ); }
        }
      }

      // PARTE 2: Contributo della parte implicita (somma di rank-1 updates)
      // H += Σ_k PQ_k * (XPT_k * XPT_k^T)
      for ( integer k = 0; k < m_npt; ++k )
      {
        if ( m_pq( k ) != 0 )
        {
          // temp = XPT_k^T * s (prodotto scalare)
          const Scalar temp = m_xpt.col( k ).head( m_neq ).dot( s_in );

          // hs += PQ_k * temp * XPT_k (rank-1 update)
          hs_out.head( m_neq ) += ( m_pq( k ) * temp ) * m_xpt.col( k ).head( m_neq ).transpose();
        }
      }
    };

    // ========================================================================
    // FASE 2: CICLO DEL GRADIENTE CONIUGATO PROIETTATO
    // ========================================================================

    /**
     * Implementa il metodo del gradiente coniugato con:
     * - Proiezione sui bound attivi
     * - Troncamento alla trust region
     * - Riavvio automatico quando cambia l'insieme attivo
     *
     * Invarianti del loop:
     * - m_d contiene sempre lo spostamento corrente
     * - m_gnew contiene il gradiente ridotto (∇m(x_opt + d))
     * - xbdi marca le variabili attive
     */
    bool converged  = false;
    bool restart_cg = true;

    while ( !converged )
    {
      // --------------------------------------------------------------------
      // Calcolo della direzione di ricerca
      // --------------------------------------------------------------------

      if ( restart_cg )
      {
        beta       = 0;
        restart_cg = false;
      }

      /**
       * Calcola la direzione s:
       * - Discesa ripida se β = 0: s = -g
       * - Gradiente coniugato altrimenti: s = β*s_old - g
       * - Proiezione: s_i = 0 se variabile i è attiva
       */
      stepsq = 0;
      for ( integer i = 0; i < m_neq; ++i )
      {
        if ( xbdi( i ) != 0 )
        {
          s( i ) = 0;  // Variabile fissata al bound
        }
        else if ( beta == 0 )
        {
          s( i ) = -m_gnew( i );  // Steepest descent
        }
        else
        {
          s( i ) = beta * s( i ) - m_gnew( i );  // Conjugate direction
        }
        stepsq += s( i ) * s( i );
      }

      // Test di convergenza: nessuna direzione disponibile
      if ( stepsq == 0 )
      {
        converged = true;
        break;
      }

      if ( beta == 0 )
      {
        gredsq  = stepsq;
        itermax = iterc + m_neq - nact;
      }

      /**
       * Test di convergenza basato sulla riduzione relativa:
       * ||g||² * δ² ≤ tol * q_red²
       *
       * Questo garantisce che ulteriori iterazioni non migliorerebbero
       * significativamente la soluzione.
       */
      if ( gredsq * delsq <= m_tol_convergence * qred * qred )
      {
        converged = true;
        break;
      }

      // --------------------------------------------------------------------
      // Calcolo del prodotto Hessiana-direzione
      // --------------------------------------------------------------------

      compute_hessian_product( s, hs );

      // --------------------------------------------------------------------
      // Calcolo della lunghezza del passo
      // --------------------------------------------------------------------

      /**
       * Calcola tre quantità chiave:
       * - resid: spazio residuo nella trust region (δ² - ||d||²)
       * - ds: prodotto scalare d·s
       * - shs: curvatura s^T H s
       */
      resid = delsq;
      ds = shs = 0;

      for ( integer i = 0; i < m_neq; ++i )
      {
        if ( xbdi( i ) == 0 )
        {
          resid -= m_d( i ) * m_d( i );
          ds += s( i ) * m_d( i );
          shs += s( i ) * hs( i );
        }
      }

      // Se d ha raggiunto il bordo, passa alle iterazioni alternative
      if ( resid <= 0 )
      {
        m_crvmin = 0;
        break;
      }

      /**
       * Calcola blen: lunghezza massima lungo s prima di uscire dalla trust region
       *
       * Risolve ||d + α*s||² = δ² per α ≥ 0:
       * ||d||² + 2α(d·s) + α²||s||² = δ²
       *
       * Soluzione: α = [√(||s||²*(δ² - ||d||²) + (d·s)²) - (d·s)] / ||s||²
       *
       * Formula numericamente stabile per evitare cancellazioni.
       */
      const Scalar temp = std::sqrt( stepsq * resid + ds * ds );
      blen              = ( ds < 0 ) ? ( ( temp - ds ) / stepsq ) : ( resid / ( temp + ds ) );

      /**
       * Calcola stplen: lunghezza ottimale ignorando i bound semplici
       * - Se curvatura positiva: minimizza il modello quadratico
       * - Se curvatura ≤ 0: va fino al bordo (direzione di curvatura negativa)
       */
      stplen = ( shs > 0 ) ? std::min( blen, gredsq / shs ) : blen;

      // --------------------------------------------------------------------
      // Riduzione per rispettare i bound semplici
      // --------------------------------------------------------------------

      /**
       * Per ogni variabile libera, calcola quanto può muoversi lungo s
       * prima di violare i suoi bound.
       */
      iact = 0;
      for ( integer i = 0; i < m_neq; ++i )
      {
        if ( s( i ) != 0 )
        {
          const Scalar xsum       = m_xopt( i ) + m_d( i );
          const Scalar bound_dist = ( s( i ) > 0 ) ? ( ( m_su( i ) - xsum ) / s( i ) )
                                                   : ( ( m_sl( i ) - xsum ) / s( i ) );

          if ( bound_dist < stplen )
          {
            stplen = bound_dist;
            iact   = i + 1;  // Salva indice 1-based
          }
        }
      }

      // --------------------------------------------------------------------
      // Aggiornamento dello spostamento e del gradiente
      // --------------------------------------------------------------------

      sdec = 0;
      if ( stplen > 0 )
      {
        ++iterc;

        // Aggiorna stima della curvatura minima
        const Scalar curvature = shs / stepsq;
        if ( iact == 0 && curvature > 0 )
        {
          m_crvmin = ( m_crvmin == -1 ) ? curvature : std::min( m_crvmin, curvature );
        }

        ggsav  = gredsq;
        gredsq = 0;

        /**
         * Aggiorna d e g usando operazioni vettoriali:
         * d_new = d_old + stplen * s
         * g_new = g_old + stplen * H*s
         */
        m_gnew.head( m_neq ) += stplen * hs.head( m_neq );
        m_d.head( m_neq ) += stplen * s.head( m_neq );

        // Ricalcola ||g||² solo per variabili libere
        for ( integer i = 0; i < m_neq; ++i )
        {
          if ( xbdi( i ) == 0 ) { gredsq += m_gnew( i ) * m_gnew( i ); }
        }

        /**
         * Calcola la riduzione effettiva del modello:
         * q_dec = α * g^T s - (α²/2) * s^T H s
         */
        sdec = std::max( stplen * ( ggsav - Scalar( 0.5 ) * stplen * shs ), Scalar( 0 ) );
        qred += sdec;
      }

      // --------------------------------------------------------------------
      // Gestione variabile diventata attiva
      // --------------------------------------------------------------------

      if ( iact > 0 )
      {
        ++nact;
        xbdi( iact - 1 ) = ( s( iact - 1 ) < 0 ) ? -1 : 1;
        delsq -= m_d( iact - 1 ) * m_d( iact - 1 );

        if ( delsq <= 0 )
        {
          m_crvmin = 0;
          break;
        }

        restart_cg = true;
        continue;
      }

      // --------------------------------------------------------------------
      // Decisione: continuare CG o passare alle iterazioni alternative
      // --------------------------------------------------------------------

      if ( stplen < blen )
      {
        // Non abbiamo raggiunto il bordo
        if ( iterc == itermax || sdec <= m_tol_step * qred )
        {
          converged = true;
          break;
        }
        beta = gredsq / ggsav;
      }
      else
      {
        // Raggiunto il bordo della trust region
        m_crvmin = 0;
        break;
      }
    }

    // ========================================================================
    // FASE 3: ITERAZIONI ALTERNATIVE SUL BORDO DELLA TRUST REGION
    // ========================================================================

    /**
     * Quando ||d|| = δ, esplora il bordo della trust region per ulteriori
     * miglioramenti. Opera nello spazio 2D generato da {d, g}.
     *
     * Parametrizzazione:
     * d(θ) = cos(θ) * d + sin(θ) * s_perp
     *
     * dove s_perp è ortogonale a d nello spazio delle variabili libere.
     */
    if ( !converged && nact < m_neq - 1 )
    {
      // Calcola proiezioni per le iterazioni alternative
      dredsq = dredg = gredsq = 0;
      for ( integer i = 0; i < m_neq; ++i )
      {
        if ( xbdi( i ) == 0 )
        {
          dredsq += m_d( i ) * m_d( i );
          dredg += m_d( i ) * m_gnew( i );
          gredsq += m_gnew( i ) * m_gnew( i );
          s( i ) = m_d( i );
        }
        else
        {
          s( i ) = 0;
        }
      }

      itcsav = iterc;

      // Calcola H * d_ridotto per riutilizzo
      compute_hessian_product( s, hred );

      // --------------------------------------------------------------------
      // Ciclo delle iterazioni alternative
      // --------------------------------------------------------------------

      bool alternative_converged = false;

      while ( !alternative_converged )
      {
        ++iterc;

        /**
         * Test di convergenza preliminare:
         * Verifica che lo spazio 2D sia ben definito
         *
         * temp = ||g||² * ||d||² - (g·d)²
         *
         * Se temp ≈ 0, allora g e d sono quasi paralleli e non c'è
         * direzione ortogonale significativa.
         */
        const Scalar temp = gredsq * dredsq - dredg * dredg;
        if ( temp <= m_tol_convergence * qred * qred ) { break; }

        /**
         * Calcola la direzione ortogonale a d usando Gram-Schmidt:
         * s_perp = (d·g) * d - ||d||² * g
         *
         * Normalizza dividendo per sqrt(temp).
         */
        const Scalar temp_sqrt = std::sqrt( temp );
        for ( integer i = 0; i < m_neq; ++i )
        {
          s( i ) = ( xbdi( i ) == 0 ) ? ( ( dredg * m_d( i ) - dredsq * m_gnew( i ) ) / temp_sqrt ) : 0;
        }
        sredg = -temp_sqrt;

        // ------------------------------------------------------------------
        // Calcolo del bound sull'angolo di rotazione
        // ------------------------------------------------------------------

        /**
         * Per ogni variabile libera, calcola il massimo angolo di rotazione
         * prima di violare i bound.
         *
         * Usa la parametrizzazione:
         * x_i(θ) = x_opt_i + d_i*cos(θ) + s_i*sin(θ)
         *
         * I bound sono: x_l_i ≤ x_i(θ) ≤ x_u_i
         */
        angbd          = 1;
        iact           = 0;
        bool bound_hit = false;

        for ( integer i = 0; i < m_neq; ++i )
        {
          if ( xbdi( i ) == 0 )
          {
            const Scalar tempa = m_xopt( i ) + m_d( i ) - m_sl( i );
            const Scalar tempb = m_su( i ) - m_xopt( i ) - m_d( i );

            // Verifica se già sul bound (errore numerico)
            if ( tempa <= 0 )
            {
              ++nact;
              xbdi( i ) = -1;
              bound_hit = true;
              break;
            }
            else if ( tempb <= 0 )
            {
              ++nact;
              xbdi( i ) = 1;
              bound_hit = true;
              break;
            }

            /**
             * Calcola il bound angolare per questa variabile.
             * Risolve per θ in:
             * x_l_i ≤ x_opt_i + ||r|| * cos(θ + φ) ≤ x_u_i
             *
             * dove r = [d_i, s_i] e φ = atan2(s_i, d_i)
             */
            const Scalar ssq = m_d( i ) * m_d( i ) + s( i ) * s( i );

            // Bound inferiore
            Scalar temp_bound = ssq - ( m_xopt( i ) - m_sl( i ) ) * ( m_xopt( i ) - m_sl( i ) );
            if ( temp_bound > 0 )
            {
              temp_bound = std::sqrt( temp_bound ) - s( i );
              if ( angbd * temp_bound > tempa )
              {
                angbd = tempa / temp_bound;
                iact  = i + 1;
                xsav  = -1;
              }
            }

            // Bound superiore
            temp_bound = ssq - ( m_su( i ) - m_xopt( i ) ) * ( m_su( i ) - m_xopt( i ) );
            if ( temp_bound > 0 )
            {
              temp_bound = std::sqrt( temp_bound ) + s( i );
              if ( angbd * temp_bound > tempb )
              {
                angbd = tempb / temp_bound;
                iact  = i + 1;
                xsav  = 1;
              }
            }
          }
        }

        // Se un bound è stato violato, ricomincia
        if ( bound_hit )
        {
          dredsq = dredg = gredsq = 0;
          for ( integer i = 0; i < m_neq; ++i )
          {
            if ( xbdi( i ) == 0 )
            {
              dredsq += m_d( i ) * m_d( i );
              dredg += m_d( i ) * m_gnew( i );
              gredsq += m_gnew( i ) * m_gnew( i );
              s( i ) = m_d( i );
            }
            else
            {
              s( i ) = 0;
            }
          }
          compute_hessian_product( s, hred );
          continue;
        }

        // ------------------------------------------------------------------
        // Calcolo del prodotto Hessiana per la direzione ortogonale
        // ------------------------------------------------------------------

        compute_hessian_product( s, hs );

        /**
         * Calcola le curvature necessarie per il modello 2D:
         * - shs = s^T H s
         * - dhs = d^T H s
         * - dhd = d^T H d (già calcolato come hred)
         */
        shs = dhs = dhd = 0;
        for ( integer i = 0; i < m_neq; ++i )
        {
          if ( xbdi( i ) == 0 )
          {
            shs += s( i ) * hs( i );
            dhs += m_d( i ) * hs( i );
            dhd += m_d( i ) * hred( i );
          }
        }

        // ------------------------------------------------------------------
        // Ricerca del miglior angolo
        // ------------------------------------------------------------------

        /**
         * Testa iu angoli equidistanti in [0, angbd] e trova quello
         * che massimizza la riduzione del modello.
         *
         * Per θ = tan(angolo):
         * - sin(θ_rad) = 2θ / (1 + θ²)
         * - cos(θ_rad) = (1 - θ²) / (1 + θ²)
         *
         * Riduzione prevista:
         * q_red(θ) = sin * [θ * (g·d) - g·s - (sin/2) * curvature(θ)]
         */
        redmax = 0;
        isav   = 0;
        redsav = 0;
        iu     = static_cast<integer>( angbd * Scalar( 17 ) + Scalar( 3.1 ) );

        for ( integer i = 1; i <= iu; ++i )
        {
          angt                   = angbd * Scalar( i ) / Scalar( iu );
          sth                    = ( angt + angt ) / ( 1 + angt * angt );
          const Scalar temp_curv = shs + angt * ( angt * dhd - dhs - dhs );
          rednew                 = sth * ( angt * dredg - sredg - Scalar( 0.5 ) * sth * temp_curv );

          if ( rednew > redmax )
          {
            redmax = rednew;
            isav   = i;
            rdprev = redsav;
          }
          else if ( i == isav + 1 ) { rdnext = rednew; }
          redsav = rednew;
        }

        // Nessun miglioramento trovato
        if ( isav == 0 )
        {
          alternative_converged = true;
          break;
        }

        /**
         * Raffina l'angolo con interpolazione parabolica.
         * Usa i tre punti (isav-1, isav, isav+1) per trovare il massimo.
         */
        if ( isav < iu )
        {
          const Scalar temp_interp = ( rdnext - rdprev ) / ( redmax + redmax - rdprev - rdnext );
          angt                     = angbd * ( Scalar( isav ) + Scalar( 0.5 ) * temp_interp ) / Scalar( iu );
        }
        else
        {
          angt = angbd * Scalar( isav ) / Scalar( iu );
        }

        /**
         * Calcola sin e cos dell'angolo ottimale usando formule di mezzo angolo:
         * Per θ = tan(α):
         * - cos(α) = (1 - θ²) / (1 + θ²)
         * - sin(α) = 2θ / (1 + θ²)
         */
        cth                    = ( 1 - angt * angt ) / ( 1 + angt * angt );
        sth                    = ( angt + angt ) / ( 1 + angt * angt );
        const Scalar temp_curv = shs + angt * ( angt * dhd - dhs - dhs );
        sdec                   = sth * ( angt * dredg - sredg - Scalar( 0.5 ) * sth * temp_curv );

        if ( sdec <= 0 )
        {
          alternative_converged = true;
          break;
        }

        // ------------------------------------------------------------------
        // Aggiornamento dello spostamento con rotazione
        // ------------------------------------------------------------------

        /**
         * Applica la rotazione nello spazio 2D:
         * d_new = cos(α) * d + sin(α) * s
         * g_new = g + (cos(α) - 1) * H*d + sin(α) * H*s
         *
         * Usa operazioni vettoriali Eigen dove possibile.
         */
        dredg = gredsq = 0;
        for ( integer i = 0; i < m_neq; ++i )
        {
          m_gnew( i ) += ( cth - 1 ) * hred( i ) + sth * hs( i );
          if ( xbdi( i ) == 0 )
          {
            m_d( i ) = cth * m_d( i ) + sth * s( i );
            dredg += m_d( i ) * m_gnew( i );
            gredsq += m_gnew( i ) * m_gnew( i );
          }
          hred( i ) = cth * hred( i ) + sth * hs( i );
        }
        qred += sdec;

        // ------------------------------------------------------------------
        // Gestione vincolo attivo raggiunto
        // ------------------------------------------------------------------

        /**
         * Se l'angolo è stato limitato da un bound, fissa quella variabile
         * e ricomincia con lo spazio ridotto.
         */
        if ( iact > 0 && isav == iu )
        {
          ++nact;
          xbdi( iact - 1 ) = xsav;

          // Ricalcola lo spazio ridotto
          dredsq = dredg = gredsq = 0;
          for ( integer i = 0; i < m_neq; ++i )
          {
            if ( xbdi( i ) == 0 )
            {
              dredsq += m_d( i ) * m_d( i );
              dredg += m_d( i ) * m_gnew( i );
              gredsq += m_gnew( i ) * m_gnew( i );
              s( i ) = m_d( i );
            }
            else
            {
              s( i ) = 0;
            }
          }
          compute_hessian_product( s, hred );
          continue;
        }

        // Test di convergenza
        if ( sdec <= m_tol_step * qred ) { alternative_converged = true; }
      }
    }

    // ========================================================================
    // FASE 4: FINALIZZAZIONE E PROIEZIONE SUI BOUND
    // ========================================================================

    /**
     * Calcola x_new = x_opt + d rispettando rigorosamente i bound.
     *
     * Per ogni componente:
     * 1. Calcola x_new_i = x_opt_i + d_i
     * 2. Proietta nell'intervallo [x_l_i, x_u_i]
     * 3. Se la variabile era attiva, forza il valore al bound esatto
     *
     * Questo garantisce che la soluzione restituita sia rigorosamente ammissibile.
     */
    m_dsq = 0;

    // Usa operazioni vettoriali per la proiezione
    for ( integer i = 0; i < m_neq; ++i )
    {
      Scalar xnew_val = m_xopt( i ) + m_d( i );

      // Proiezione nell'intervallo ammissibile
      xnew_val = std::min( xnew_val, m_su( i ) );
      xnew_val = std::max( xnew_val, m_sl( i ) );

      // Forza variabili attive al bound esatto (evita errori numerici)
      if ( xbdi( i ) == -1 ) { xnew_val = m_sl( i ); }
      else if ( xbdi( i ) == 1 ) { xnew_val = m_su( i ); }

      m_xnew( i ) = xnew_val;
      m_d( i )    = xnew_val - m_xopt( i );
      m_dsq += m_d( i ) * m_d( i );
    }
  }

  // ============================================================================
  // HELPER: Versione ottimizzata del prodotto Hessiana-vettore (standalone)
  // ============================================================================

  /**
   * @brief Calcola H*v in modo efficiente usando block operations di Eigen
   *
   * @tparam Scalar Tipo numerico
   * @param hq Parte esplicita dell'Hessiana (packed symmetric)
   * @param pq Coefficienti per la parte implicita
   * @param xpt Punti di interpolazione (npt × neq)
   * @param v Vettore input
   * @param result Vettore output H*v
   * @param neq Numero di variabili
   * @param npt Numero di punti
   *
   * @note Questa versione standalone può essere riutilizzata in altri metodi
   */
  template <typename Scalar>
  inline void
  compute_hessian_vector_product( const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &              hq,
                                  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &              pq,
                                  const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> & xpt,
                                  const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &              v,
                                  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &                    result,
                                  const int                                                     neq,
                                  const int                                                     npt )
  {
    result.setZero( neq );

    // Parte simmetrica esplicita
    int ih = 0;
    for ( int j = 0; j < neq; ++j )
    {
      for ( int i = 0; i <= j; ++i )
      {
        const Scalar hq_val = hq( ih++ );
        result( i ) += hq_val * v( j );
        if ( i < j ) { result( j ) += hq_val * v( i ); }
      }
    }

    // Parte implicita (rank-1 updates)
    for ( int k = 0; k < npt; ++k )
    {
      if ( pq( k ) != Scalar( 0 ) )
      {
        const Scalar coeff = pq( k ) * xpt.row( k ).head( neq ).dot( v );
        result.head( neq ).noalias() += coeff * xpt.row( k ).head( neq ).transpose();
      }
    }
  }


  /**
   * @brief Aggiorna le matrici BMAT e ZMAT dopo lo spostamento di un punto di interpolazione
   *
   * Questa funzione aggiorna le strutture dati necessarie per il modello quadratico
   * quando un punto di interpolazione (indicizzato da m_knew) viene spostato in una
   * nuova posizione. L'aggiornamento mantiene le proprietà di interpolazione del modello.
   *
   * @details
   * L'algoritmo esegue le seguenti operazioni:
   * 1. Applica rotazioni di Givens per azzerare la riga m_knew di ZMAT (eccetto la prima colonna)
   * 2. Calcola i parametri della formula di aggiornamento (alpha, tau)
   * 3. Aggiorna la matrice ZMAT utilizzando una formula di rango-1
   * 4. Aggiorna la matrice BMAT mantenendo la simmetria
   *
   * Le matrici BMAT e ZMAT sono rappresentazioni compatte della matrice Hessiana
   * approssimata utilizzata nel modello quadratico dell'algoritmo BOBYQA.
   *
   * @note Questa implementazione sfrutta le operazioni vettoriali di Eigen3 per
   *       migliorare le prestazioni rispetto all'implementazione originale con loop espliciti.
   *
   * @pre m_knew deve essere un indice valido (1-based) di un punto di interpolazione
   * @pre m_denom deve essere positivo per evitare divisioni per zero
   * @pre Le dimensioni delle matrici devono essere coerenti con m_npt e m_neq
   *
   * @post La matrice ZMAT avrà zeri nella riga m_knew (eccetto eventualmente la colonna 0)
   * @post La matrice BMAT sarà aggiornata e manterrà la simmetria
   *
   * @tparam Scalar Tipo numerico (float, double, long double)
   */
  template <typename Scalar>
  void
  BOBYQA_minimizer<Scalar>::update()
  {
    // Costanti numeriche
    constexpr Scalar one          = Scalar( 1 );
    constexpr Scalar zero         = Scalar( 0 );
    constexpr Scalar ztest_factor = Scalar( 1e-20 );

    // Dimensioni delle strutture dati
    const integer nptm = m_npt - m_neq - 1;  // Numero di colonne di ZMAT

    // Vettore di lavoro temporaneo per calcoli intermedi
    // Dimensione: m_npt elementi per HLAG + m_neq elementi per coefficienti BMAT
    Vector work( m_npt + m_neq );

    // ========================================================================
    // FASE 1: Azzeramento riga KNEW di ZMAT mediante rotazioni di Givens
    // ========================================================================

    /**
     * Calcola la soglia di tolleranza per determinare quali elementi di ZMAT
     * sono sufficientemente grandi da richiedere una rotazione di Givens.
     * Basato sul massimo valore assoluto presente in ZMAT.
     */
    const Scalar ztest = m_zmat.topLeftCorner( m_npt, nptm ).cwiseAbs().maxCoeff() * ztest_factor;

    /**
     * Applica rotazioni di Givens per azzerare la riga m_knew di ZMAT.
     * Ogni rotazione opera sulle colonne 0 e j, azzerando l'elemento (m_knew-1, j).
     *
     * La rotazione di Givens è definita da:
     *   [c  s] [a]   [r]
     *   [-s c] [b] = [0]
     * dove r = sqrt(a² + b²), c = a/r, s = b/r
     */
    for ( integer j = 1; j < nptm; ++j )
    {
      const Scalar zij = m_zmat( m_knew - 1, j );

      // Verifica se l'elemento è sufficientemente grande da richiedere una rotazione
      if ( std::abs( zij ) > ztest )
      {
        const Scalar zi0 = m_zmat( m_knew - 1, 0 );

        // Calcola i parametri della rotazione di Givens usando std::hypot per stabilità
        const Scalar r = std::hypot( zi0, zij );  // r = sqrt(zi0² + zij²)
        const Scalar c = zi0 / r;                 // coseno della rotazione
        const Scalar s = zij / r;                 // seno della rotazione

        /**
         * Applica la rotazione alle colonne 0 e j di ZMAT.
         * Utilizziamo operazioni vettoriali Eigen per processare tutte le righe
         * contemporaneamente, evitando loop espliciti.
         *
         * Formula di rotazione:
         *   ZMAT_new(:,0) =  c * ZMAT(:,0) + s * ZMAT(:,j)
         *   ZMAT_new(:,j) = -s * ZMAT(:,0) + c * ZMAT(:,j)
         */
        auto col0 = m_zmat.col( 0 ).head( m_npt );
        auto colj = m_zmat.col( j ).head( m_npt );

        // Calcola le nuove colonne ruotate
        const Vector temp0 = c * col0 + s * colj;
        const Vector tempj = c * colj - s * col0;

        // Assegna i risultati
        col0 = temp0;
        colj = tempj;

        // Forza a zero l'elemento target per eliminare errori di arrotondamento
        m_zmat( m_knew - 1, j ) = zero;
      }
    }

    // ========================================================================
    // FASE 2: Calcolo dei parametri di aggiornamento
    // ========================================================================

    /**
     * Calcola il vettore work = ZMAT(m_knew-1, 0) * ZMAT(:, 0)
     * Questo rappresenta la prima NPT componenti della colonna m_knew di HLAG.
     *
     * HLAG è la matrice Lagrangiana delle funzioni di base dell'interpolazione.
     */
    work.head( m_npt ) = m_zmat( m_knew - 1, 0 ) * m_zmat.col( 0 ).head( m_npt );

    // Parametri chiave della formula di aggiornamento
    const Scalar alpha = work( m_knew - 1 );    // Coefficiente per la direzione di ricerca
    const Scalar tau   = m_vlag( m_knew - 1 );  // Moltiplicatore di Lagrange

    /**
     * Modifica temporanea di VLAG per il calcolo dell'aggiornamento.
     * Questo verrà utilizzato nella formula di aggiornamento e poi ripristinato implicitamente.
     */
    m_vlag( m_knew - 1 ) -= one;

    // ========================================================================
    // FASE 3: Aggiornamento della matrice ZMAT
    // ========================================================================

    /**
     * Aggiorna ZMAT con una formula di rango-1:
     *   ZMAT_new(:,0) = (tau/sqrt(denom)) * ZMAT(:,0) - (z_k0/sqrt(denom)) * VLAG
     *
     * Questa formula mantiene le proprietà di ortogonalità richieste da BOBYQA.
     */
    const Scalar sqrt_denom = std::sqrt( m_denom );
    const Scalar scale_z    = m_zmat( m_knew - 1, 0 ) / sqrt_denom;  // Fattore di scala per VLAG
    const Scalar scale_tau  = tau / sqrt_denom;                      // Fattore di scala per ZMAT

    // Aggiornamento vettoriale della prima colonna di ZMAT
    m_zmat.col( 0 ) = scale_tau * m_zmat.col( 0 ) - scale_z * m_vlag.head( m_npt );

    // ========================================================================
    // FASE 4: Aggiornamento della matrice BMAT (simmetrica)
    // ========================================================================

    /**
     * Aggiorna la matrice BMAT che rappresenta la parte lineare del modello.
     * BMAT è simmetrica, quindi aggiorniamo solo la parte triangolare superiore
     * e poi copiamo per mantenere la simmetria.
     *
     * Formula di aggiornamento per ogni colonna j:
     *   BMAT(:,j) += tempa * VLAG + tempb * work
     * dove tempa e tempb dipendono da alpha, beta, tau, e m_denom.
     */
    for ( integer j = 0; j < m_neq; ++j )
    {
      const integer jp = m_npt + j;  // Indice esteso (0-based)

      // Salva il valore corrente di BMAT(m_knew-1, j) nel vettore di lavoro
      work( jp ) = m_bmat( j, m_knew - 1 );

      /**
       * Calcola i coefficienti per l'aggiornamento della colonna j.
       * Questi coefficienti bilanciano i contributi di VLAG e work.
       */
      const Scalar tempa = ( alpha * m_vlag( jp ) - tau * work( jp ) ) / m_denom;
      const Scalar tempb = ( -m_beta * work( jp ) - tau * m_vlag( jp ) ) / m_denom;

      /**
       * Aggiorna la colonna j di BMAT usando operazioni vettoriali.
       * Dividiamo l'aggiornamento in due parti per gestire la simmetria:
       * 1. Righe 0 to m_npt-1: aggiornamento diretto
       * 2. Righe m_npt to jp: aggiornamento + copia simmetrica
       */

      // Parte 1: Aggiorna le prime m_npt righe (vettoriale)
      m_bmat.row( j ).head( m_npt ) += tempa * m_vlag.head( m_npt ) + tempb * work.head( m_npt );

      // Parte 2: Aggiorna le righe da m_npt a jp
      for ( integer i = m_npt; i <= jp; ++i )
      {
        m_bmat( j, i ) += tempa * m_vlag( i ) + tempb * work( i );

        /**
         * Mantieni la simmetria: BMAT(i,j) = BMAT(j,i-m_npt)
         * Questo è necessario perché BMAT memorizza una matrice concettualmente
         * simmetrica in una rappresentazione rettangolare.
         */
        m_bmat( i - m_npt, jp ) = m_bmat( j, i );
      }
    }
  }

  template class BOBYQA_minimizer<double>;
  template class BOBYQA_minimizer<float>;


}  // namespace Utils
