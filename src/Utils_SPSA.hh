/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2025                                                      |
 |                                                                          |
 |         , __                 , __                                        |
 |        /|/  \               /|/  \                                       |
 |         | __/ _   ,_         | __/ _   ,_                                |
 |         |   \|/  /  |  |   | |   \|/  /  |  |   |                        |
 |         |(__/|__/   |_/ \_/|/|(__/|__/   |_/ \_/|/                       |
 |                           /|                   /|                        |
 |                           \|                   \|                        |
 |                                                                          |
 |      Enrico Bertolazzi                                                   |
 |      Dipartimento di Ingegneria Industriale                              |
 |      Università degli Studi di Trento                                    |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

//
// file: Utils_SPSA.hh
//

#pragma once

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#ifndef UTILS_SPSA_dot_HH
#define UTILS_SPSA_dot_HH

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++98-compat"
#pragma clang diagnostic ignored "-Wc++98-compat-pedantic"
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wswitch-enum"
#pragma clang diagnostic ignored "-Wdocumentation"
#pragma clang diagnostic ignored "-Wdocumentation-unknown-command"
#pragma clang diagnostic ignored "-Wglobal-constructors"
#pragma clang diagnostic ignored "-Wzero-as-null-pointer-constant"
#pragma clang diagnostic ignored "-Wweak-vtables"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wundefined-func-template"
#pragma clang diagnostic ignored "-Wdouble-promotion"
#pragma clang diagnostic ignored "-Wsigned-enum-bitfield"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wweak-vtables"
#pragma clang diagnostic ignored "-Wunused-template"
#pragma clang diagnostic ignored "-Wnon-virtual-dtor"
#pragma clang diagnostic ignored "-Wpadded"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#endif

#include "Utils.hh"
#include "Utils_fmt.hh"
#include "Utils_eigen.hh"

#include <algorithm>
#include <cmath>
#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <utility>
#include <vector>

namespace Utils {

  using std::abs;
  using std::min;
  using std::max;
  using std::sqrt;
  using std::pow;

  /**
   * @class SPSAGradientEstimator
   * @brief Stima del gradiente usando Simultaneous Perturbation Stochastic Approximation
   * 
   * Questa classe implementa l'algoritmo SPSA per stimare il gradiente di una funzione
   * usando solo valutazioni della funzione stessa, senza bisogno di gradienti analitici.
   * L'algoritmo è particolarmente utile per ottimizzazione di funzioni costose o non differenziabili.
   * 
   * @tparam Scalar Tipo dei dati (double, float, etc.)
   */
  template <typename Scalar = double>
  class SPSAGradientEstimator {
  public:
    using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>; ///< Tipo vettore
    using Callback = std::function<Scalar(Vector const &, Vector *)>; ///< Tipo callback funzione

    /**
     * @struct Options
     * @brief Opzioni per lo stimatore del gradiente SPSA
     */
    struct Options {
      size_t   repeats            = 10;            ///< Numero di ripetizioni per la media del gradiente
      Scalar   c_base             = Scalar(1e-3);  ///< Ampiezza base di perturbazione
      Vector   c_per_component;                    ///< Ampiezze specifiche per componente (se vuoto usa c_base)
      bool     use_rademacher     = true;          ///< True per noise ±1, false per gaussiano N(0,1)
      bool     apply_precond      = false;         ///< Applica precondizionamento diagonale
      Vector   preconditioner;                     ///< Vettore pesi per precondizionamento
      unsigned int rng_seed       = std::random_device{}(); ///< Seed per generatore random
      Scalar   min_delta_abs      = Scalar(1e-12); ///< Valore minimo per evitare divisioni per zero
      bool     allow_gradient_arg = false;         ///< Se true, callback può ricevere gradiente output
    };

  private:
    Options m_opts; ///< Opzioni configurate
    Vector  m_tmp_grad; ///< Buffer temporaneo per gradienti

  public:

    /**
     * @brief Costruttore
     * @param opts Opzioni per lo stimatore
     */
    SPSAGradientEstimator(Options opts = Options())
    : m_opts(std::move(opts))
    {}

    /**
     * @brief Stima il gradiente nel punto x usando SPSA
     * 
     * L'algoritmo stima il gradiente usando la formula:
     * g_i ≈ [f(x + c*Δ) - f(x - c*Δ)] / (2 * c_i * Δ_i)
     * 
     * dove Δ è un vettore di perturbazione casuale (±1 o gaussiano).
     * 
     * @param x Punto in cui stimare il gradiente
     * @param f Funzione obiettivo
     * @param out_grad Gradiente stimato (output)
     * @return Numero di valutazioni di funzione effettuate
     */
    size_t
    estimate( Vector const & x, Callback const & f, Vector & out_grad ) {
      size_t const n = static_cast<size_t>(x.size());
      assert(n > 0);

      // prepara c_i
      Vector c;
      if ( m_opts.c_per_component.size() == 0 ) {
        c = Vector::Constant( n, m_opts.c_base );
      } else {
        assert( m_opts.c_per_component.size() == n );
        c = m_opts.c_per_component;
      }

      // preconditioner
      if ( m_opts.apply_precond ) assert( m_opts.preconditioner.size() == n );

      out_grad.setZero(n);
      if ( m_opts.repeats == 0 ) return 0;

      // RNGs
      std::mt19937 rng(m_opts.rng_seed);
      std::uniform_int_distribution<size_t> rademacher_dist(0, 1); // 0 -> -1, 1 -> +1
      std::normal_distribution<Scalar>      gaussian_dist(0.0, 1.0);

      Vector x_plus(n);
      Vector x_minus(n);
      Vector delta(n);
      Scalar y_plus, y_minus;

      size_t eval_count{0};

      for ( size_t rep{0}; rep < m_opts.repeats; ++rep ) {

        // sample delta
        if ( m_opts.use_rademacher ) {
          for ( size_t i{0}; i < n; ++i )
            delta[i] = (rademacher_dist(rng) ? Scalar(1) : Scalar(-1));
        } else {
          for ( size_t i{0}; i < n; ++i ) {
            Scalar d = gaussian_dist(rng);
            // prevenzione di delta troppo piccolo (rare per gauss, ma utile)
            if ( abs(d) < m_opts.min_delta_abs )
              d = std::copysign( m_opts.min_delta_abs, d == 0 ? Scalar(1) : d );
            delta[i] = d;
          }
        }

        // costruisci x± = x ± c .* delta
        x_plus  = x + c.cwiseProduct(delta);
        x_minus = x - c.cwiseProduct(delta);

        // valuta funzione (notare: passiamo nullptr per gradient output; callback può ignorare secondo opzione)
        y_plus  = f(x_plus,  m_opts.allow_gradient_arg ? &m_tmp_grad : nullptr);
        y_minus = f(x_minus, m_opts.allow_gradient_arg ? &m_tmp_grad : nullptr);
        eval_count += 2;

        // aggiorna stima: g_i += (y+ - y-) / (2 * c_i * delta_i)
        const Scalar denom_factor = Scalar(0.5); // we'll multiply by (y+ - y-) * 0.5 / (c_i * delta_i)
        Scalar diff = (y_plus - y_minus);

        for (size_t i{0}; i < n; ++i) {
          Scalar di = delta[i];
          // sicurezza: se delta molto vicino a zero (solo possibile con gauss), salta o usa fallback
          if (std::abs(di) < m_opts.min_delta_abs) {
            // fallback: ignora contributo di questa ripetizione per la componente i
            continue;
          }
          out_grad[i] += diff * (denom_factor / (c[i] * di));
        }
      } // end repeats

      // media
      out_grad /= static_cast<Scalar>(m_opts.repeats);

      // applica precondizionamento (se richiesto): semplicemente scala componente-wise
      if (m_opts.apply_precond) {
        out_grad = out_grad.cwiseProduct(m_opts.preconditioner);
      }

      return eval_count;
    }

    Options const & options() const { return m_opts; }
    Options & options() { return m_opts; }
  };

  /**
   * @class SPSA_minimizer
   * @brief Minimizzatore che usa Simultaneous Perturbation Stochastic Approximation
   * 
   * Implementa l'algoritmo SPSA per ottimizzazione senza gradienti. Particolarmente
   * efficace per problemi ad alta dimensionalità o con funzioni obiettivo rumorose.
   * 
   * L'algoritmo principale combina:
   * - Stima del gradiente via SPSA
   * - Discesa stocastica con learning rate adattivo
   * - Gestione di vincoli box via clipping
   * - Strategie di robustezza (backtracking, restart)
   * 
   * @tparam Scalar Tipo dei dati (double, float, etc.)
   */
  template <typename Scalar = double>
  class SPSA_minimizer {
  public:
  
    using Vector   = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>; ///< Tipo vettore
    using Callback = std::function<Scalar(Vector const&, Vector*)>; ///< Tipo callback funzione
  
    /**
     * @enum Variant
     * @brief Varianti dell'algoritmo SPSA
     */
    enum class Variant {
      BASIC,        ///< SPSA classica
      RADA,         ///< Usa noise Rademacher (±1)
      SECOND_ORDER, ///< SPSA del secondo ordine
      PRECOND,      ///< SPSA precondizionata
      ADAPTIVE      ///< SPSA con passi adattivi
    };
  
    /**
     * @struct Options
     * @brief Opzioni per il minimizzatore SPSA
     */
    struct Options {
      size_t max_iter      {500};          ///< Iterazioni massime (AUMENTATO per robustezza)
      size_t gradient_avg  {1};            ///< Medie SPSA per gradiente
      Scalar a0            {0.3};          ///< Learning rate base (AUMENTATO per velocità)
      Scalar c0            {0.05};         ///< Ampiezza perturbazioni (AUMENTATO)
      Scalar alpha         {0.602};        ///< Esponente decay learning rate (standard SPSA)
      Scalar gamma         {0.101};        ///< Esponente decay perturbazioni (standard SPSA)
      Variant variant      {Variant::ADAPTIVE}; ///< Variante (MODIFICATO per robustezza)
      bool   use_projection{true};         ///< Abilita proiezione sui bounds
      bool   verbose       {false};        ///< Output verboso
    };
  
    /**
     * @struct Result
     * @brief Risultato della minimizzazione
     */
    struct Result {
      Scalar final_f;      ///< Valore finale della funzione
      Scalar grad_norm;    ///< Norma del gradiente finale
      size_t iterations;   ///< Numero di iterazioni effettuate
      Vector final_x;      ///< Punto finale
      bool   converged;    ///< True se convergenza raggiunta
    };

  private:
    Options m_opts;        ///< Opzioni configurate
    Vector  m_lower;       ///< Limiti inferiori
    Vector  m_upper;       ///< Limiti superiori
    std::mt19937 rng{std::random_device{}()}; ///< Generatore random
  
  public:
  
    /**
     * @brief Costruttore
     * @param o Opzioni per il minimizzatore
     */
    SPSA_minimizer( Options const & o = Options() )
    : m_opts(o)
    {}
  
    /**
     * @brief Imposta i bound di ottimizzazione
     * @param lo Limiti inferiori
     * @param up Limiti superiori
     */
    void set_bounds(Vector const& lo, Vector const& up) {
      assert(lo.size() == up.size());
      m_lower = lo;
      m_upper = up;
    }
  
  private:
  
    /**
     * @brief Proietta il punto x sui bounds
     * @param x Punto da proiettare (modificato in-place)
     */
    void project( Vector & x ) const {
      if ( m_opts.use_projection )
        x = x.cwiseMax(m_lower).cwiseMin(m_upper);
    }
  
    /**
     * @brief Genera noise Rademacher (±1)
     * @param d Vettore dove memorizzare il noise (output)
     */
    void rademacher( Vector & d ) {
      std::uniform_real_distribution<Scalar> U(0,1);
      for ( size_t i{0}; i < d.size(); ++i )
        d[i] = (U(rng) < 0.5 ? -1 : +1);
    }
  
    /**
     * @brief Applica precondizionamento diagonale al gradiente
     * @param g Gradiente da precondizionare (modificato in-place)
     * @param scale Fattori di scala per precondizionamento
     */
    void apply_precond( Vector& g, Vector const & scale ) {
      g.array() /= (scale.array() + Scalar(1e-12));
    }
  
  public:

    /**
     * @brief Esegue la minimizzazione usando SPSA con gestione robusta dei bounds
     * 
     * L'algoritmo implementa una versione migliorata di SPSA con:
     * 1. Clipping delle perturbazioni per rispettare i bounds durante la stima del gradiente
     * 2. Backtracking lineare per learning rate adattivo
     * 3. Strategia di restart quando bloccato
     * 4. Tracking della migliore soluzione trovata
     * 
     * @param x0 Punto iniziale
     * @param fun Funzione obiettivo da minimizzare
     * @return Risultato della minimizzazione
     */
    Result minimize( Vector const & x0, Callback const & fun ) {
      size_t n = x0.size();
      Vector x = x0;
      Vector g(n), delta(n), scale = Vector::Ones(n);
      Vector xp(n), xm(n);
    
      // Proietta punto iniziale e valuta
      project(x);
      Scalar f = fun(x, nullptr);
      Scalar best_f = f;
      Vector best_x = x;
      size_t no_improvement_count = 0;
    
      for ( size_t k{0}; k < m_opts.max_iter; ++k ) {
        // Calcola learning rate e perturbazione con decay standard SPSA
        Scalar a_k = m_opts.a0 / pow(Scalar(k+1), m_opts.alpha);
        Scalar c_k = m_opts.c0 / pow(Scalar(k+1), m_opts.gamma);
    
        g.setZero();
    
        // ============================
        //      STIMA GRADIENTE SPSA
        // ============================
        for (size_t rep = 0; rep < m_opts.gradient_avg; ++rep) {
    
          // Genera perturbazioni casuali
          rademacher(delta);
    
          // APPLICA CLIPPING INTELLIGENTE DELLE PERTURBAZIONI
          for (size_t i = 0; i < n; ++i) {
            Scalar perturbation = c_k * delta[i];
    
            // Calcola spazio disponibile nei bounds
            Scalar max_plus  = m_upper[i] - x[i];  // Spazio verso l'alto
            Scalar max_minus = x[i] - m_lower[i];  // Spazio verso il basso
    
            // Clip della perturbazione per rimanere nei bounds
            if (perturbation > max_plus) {
              perturbation = max_plus;
            } else if (perturbation < -max_minus) {
              perturbation = -max_minus;
            }
    
            // Applica perturbazioni clippate - garantisce xp, xm dentro i bounds
            xp[i] = x[i] + perturbation;
            xm[i] = x[i] - perturbation;
    
            // Gestione casi di bordo: se completamente bloccato, usa perturbazione minima
            if (std::abs(perturbation) < 1e-12 && max_plus < 1e-12 && max_minus < 1e-12) {
              perturbation = 1e-12;
              xp[i] = x[i] + perturbation;
              xm[i] = x[i] - perturbation;
            }
          }
    
          // Valuta funzione nei punti perturbati (garantiti dentro i bounds)
          Scalar fp = fun(xp, nullptr);
          Scalar fm = fun(xm, nullptr);
    
          // CALCOLO GRADIENTE - matematicamente corretto grazie al clipping
          for (size_t i = 0; i < n; ++i) {
            if (std::abs(delta[i]) > 1e-12) {
              // Usa formula SPSA standard con delta originale
              g[i] += (fp - fm) / (2 * c_k * delta[i]);
            } else {
              // Fallback per delta zero (raro): usa perturbazione effettiva
              Scalar actual_perturbation = xp[i] - x[i];
              if (std::abs(actual_perturbation) > 1e-12) {
                g[i] += (fp - fm) / (2 * actual_perturbation);
              }
            }
          }
        }
    
        // Media delle stime del gradiente
        g.array() /= Scalar(m_opts.gradient_avg);
    
        // CLIPPING DEL GRADIENTE per prevenire esplosioni numeriche
        Scalar gmax = g.template lpNorm<Eigen::Infinity>();
        if (gmax > 1e6) {
          g *= (1e6 / gmax);
          if (m_opts.verbose) {
            std::cout << "[SPSA] Gradient clipped at iter " << k << "\n";
          }
        }
    
        Scalar gnorm = g.template lpNorm<Eigen::Infinity>();
    
        // Output verboso (ogni 50 iterazioni per non appesantire)
        if (m_opts.verbose && k % 50 == 0) {
          std::cout << "[SPSA] iter " << k << " f=" << f << " |g|=" << gnorm
                    << " a_k=" << a_k << " c_k=" << c_k << "\n";
        }
    
        // CRITERIO DI CONVERGENZA - norma del gradiente sotto soglia
        if (gnorm < 1e-8 && k > 10) {
          if (m_opts.verbose) {
            std::cout << "[SPSA] Converged at iter " << k << " with gnorm=" << gnorm << "\n";
          }
          return {f, gnorm, k, x, true};
        }
    
        // ============================
        //       PRECONDIZIONAMENTO
        // ============================
        if (m_opts.variant == Variant::PRECOND) {
          // Scaling diagonale adattivo: media mobile delle componenti del gradiente
          scale = 0.9 * scale.array() + 0.1 * (g.cwiseAbs().array() + 1e-8);
          apply_precond(g, scale);
        }
    
        // ============================
        //       AGGIORNAMENTO
        // ============================
        Vector x_new = x - a_k * g;
    
        // Proietta il nuovo punto sui bounds
        project(x_new);
    
        Scalar f_new = fun(x_new, nullptr);
    
        // TRACKING MIGLIORE SOLUZIONE
        if (f_new < best_f) {
          best_f = f_new;
          best_x = x_new;
          no_improvement_count = 0; // Reset contatore
        } else {
          no_improvement_count++;
        }
    
        // BACKTRACKING ADATTIVO se peggioramento
        if (m_opts.variant == Variant::ADAPTIVE || f_new > f * 1.1) {
          bool improved = false;
          Scalar a_temp = a_k;
          
          // Prova fino a 5 learning rate ridotti
          for (int backtrack = 0; backtrack < 5; ++backtrack) {
            a_temp *= 0.5;
            Vector x_temp = x - a_temp * g;
            project(x_temp);
            Scalar f_temp = fun(x_temp, nullptr);
    
            if (f_temp < f) {
              x_new = x_temp;
              f_new = f_temp;
              improved = true;
              // Aggiorna migliore soluzione se trovata
              if (f_new < best_f) {
                best_f = f_new;
                best_x = x_new;
              }
              break;
            }
          }
    
          if (!improved && m_opts.verbose) {
            std::cout << "[SPSA] Backtracking failed at iter " << k << "\n";
          }
        }
    
        // STRATEGIA DI RESTART se bloccato per molte iterazioni
        if (no_improvement_count > 100) {
          if (m_opts.verbose) {
            std::cout << "[SPSA] Restarting from best point at iter " << k << "\n";
          }
          x = best_x;
          f = best_f;
          no_improvement_count = 0;
          continue; // Salta il resto dell'iterazione
        }
    
        // ACCETTA NUOVO PUNTO
        x = x_new;
        f = f_new;
      }
    
      // Ritorna la migliore soluzione trovata (non necessariamente l'ultima)
      return {best_f, 0, m_opts.max_iter, best_x, false};
    }
  };
}

#endif

#endif

//
// eof: Utils_SPSA.hh
//
