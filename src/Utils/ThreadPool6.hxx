/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2020                                                      |
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
 |      Universita` degli Studi di Trento                                   |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

//
// file: ThreadPool6.hxx
//

#include "3rd/BS_thread_pool.hpp"

namespace Utils {

  /*!
   * \addtogroup THREAD
   * @{
   */

  /*\
   |   _____ _                        _ ____             _
   |  |_   _| |__  _ __ ___  __ _  __| |  _ \ ___   ___ | |
   |    | | | '_ \| '__/ _ \/ _` |/ _` | |_) / _ \ / _ \| |
   |    | | | | | | | |  __/ (_| | (_| |  __/ (_) | (_) | |
   |    |_| |_| |_|_|  \___|\__,_|\__,_|_|   \___/ \___/|_|
  \*/
  //!
  //! \brief A thread pool for concurrent task execution with worker management.
  //!
  //! The `ThreadPool6` class provides an implementation of a thread pool that
  //! allows for efficient concurrent execution of tasks. This class manages a
  //! collection of worker threads that can be dynamically resized, and utilizes
  //! a semaphore-based mechanism for worker synchronization.
  //!
  //! This class extends `ThreadPoolBase` and supports task execution, joining,
  //! and resizing of worker threads.
  //!
  class ThreadPool6 : public ThreadPoolBase {

    BS::pause_thread_pool m_pool;

  public:

    //!
    //! \brief Constructs a new ThreadPool5 instance with a specified number of threads.
    //!
    //! \param nthread The number of threads to create in the pool. Defaults to the maximum hardware threads available.
    //!
    ThreadPool6(
      unsigned nthread = std::max(
        unsigned(1),
        unsigned(std::thread::hardware_concurrency()-1)
      )
    ) : m_pool( nthread ) {}

    //!
    //! \brief Destructor for the ThreadPool5 class.
    //!
    //! Ensures all workers are stopped and joined before destruction.
    //!
    virtual ~ThreadPool6() {}

    //!
    //! \brief Executes a task and assigns it to an available worker.
    //!
    //! \param fun The function to be executed as a task.
    //!
    void exec( std::function<void()> && fun ) override { m_pool.detach_task( fun ); }

    //!
    //! \brief Waits for all tasks to be completed.
    //!
    void wait() override { m_pool.wait(); }

    //!
    //! \brief Joins all worker threads and ensures they are stopped.
    //!
    void join() override { m_pool.purge(); }

    //!
    //! \brief Gets the current number of threads in the pool.
    //!
    //! \return The number of threads in the pool.
    //!
    unsigned thread_count() const override { return unsigned( m_pool.get_thread_count() ); }

    //!
    //! \brief Resizes the thread pool to a new number of workers.
    //!
    //! \param numThreads The new number of threads to maintain in the pool.
    //!
    void resize( unsigned numThreads ) override { m_pool.reset( numThreads ); }

    //!
    //! \brief Gets the name of the thread pool implementation.
    //!
    //! \return A constant character pointer to the name of the thread pool.
    //!
    char const * name() const override { return "ThreadPool6"; }

    //!
    //! \brief Starts all workers in the pool.
    //!
    void start() { m_pool.unpause(); }

    //!
    //! \brief Stops all workers in the pool.
    //!
    void stop() { m_pool.pause(); }
  };

  /*! @} */

}

//
// eof: ThreadPool6.hxx
//
