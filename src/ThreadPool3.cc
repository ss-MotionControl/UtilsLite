/*--------------------------------------------------------------------------*\
 |                                                                          |
 |  Copyright (C) 2022                                                      |
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
 |      Via Sommarive 9, I-38123 Povo, Trento, Italy                        |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

//
// file: ThreadPool3.cc
//

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#if defined(__llvm__) || defined(__clang__)
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wduplicate-enum"
#endif

#include "Utils.hh"
#include "Utils_fmt.hh"

namespace Utils {

  ThreadPool3::ThreadPool3( unsigned thread_count, unsigned queue_capacity )
  : m_work_queue( queue_capacity == 0 ? std::max( 10 * (thread_count+1), unsigned(4096) ) : queue_capacity )
  {
    create_workers( thread_count );
  }

  //
  // https://stackoverflow.com/questions/48936591/is-the-performance-of-notify-one-really-this-bad
  //

  void
  ThreadPool3::push_task( TaskData * task ) {
    {
      std::unique_lock<std::mutex> lock( m_queue_push_mutex );
      ++m_push_waiting;
      while ( m_work_queue.is_full() ) m_queue_push_cv.wait( m_queue_push_mutex );
      //-----------------
      m_queue_spin.lock();
      m_work_queue.push( task );
      --m_push_waiting;
      m_queue_spin.unlock();
      //-----------------
    }

    // push done
    if ( m_pop_waiting > 0 ) m_queue_pop_cv.notify_one();
    if ( m_push_waiting > 0 ) {
      m_queue_spin.lock();
      if ( !m_work_queue.is_full() ) m_queue_push_cv.notify_one();
      m_queue_spin.unlock();
    }
  }

  tp::Queue::TaskData *
  ThreadPool3::pop_task() {
    TaskData * task{ nullptr };
    {
      std::unique_lock<std::mutex> lock( m_queue_pop_mutex );
      ++m_pop_waiting;
      m_queue_pop_cv.wait( m_queue_pop_mutex, [&]()->bool { return !m_work_queue.empty(); } );
      //-----------------
      m_queue_spin.lock();
      task = m_work_queue.pop();
      --m_pop_waiting;
      m_queue_spin.unlock();
      ++m_running_task; // must be incremented in the locked part
      //-----------------
    }
    
    
    if ( m_push_waiting > 0 ) m_queue_push_cv.notify_one();
    if ( m_pop_waiting  > 0 ) {
      m_queue_spin.lock();
      if ( !m_work_queue.empty() ) m_queue_pop_cv.notify_one();
      m_queue_spin.unlock();
    }
    return task;
  }

  void
  ThreadPool3::worker_thread() {
    ++m_running_thread;
    while ( !m_done ) {
      // ---------------------------- POP
      TaskData * task{ pop_task() };
      // ---------------------------- RUN
      (*task)(); // run and delete task;
      // ---------------------------- UPDATE
      --m_running_task;
    }
    --m_running_thread;
  }

  void
  ThreadPool3::create_workers( unsigned thread_count ) {
    m_worker_threads.clear();
    m_worker_threads.reserve(thread_count);
    m_done         = false;
    m_push_waiting = 0;
    m_pop_waiting  = 0;
    try {
      for ( unsigned i{0}; i<thread_count; ++i )
        m_worker_threads.emplace_back( &ThreadPool3::worker_thread, this );
    } catch(...) {
      m_done = true;
      throw;
    }
  }

  void
  ThreadPool3::join() {
    wait();
    m_done = true;
    { // send null task until all the workers stopped
      std::function<void()> null_job = [](){};
      for ( unsigned i{m_running_thread}; i > 0; --i )
        push_task( new TaskData(null_job) );
      while ( m_running_thread > 0 ) std::this_thread::yield();
    }
    m_work_queue.clear();
    for ( std::thread & w : m_worker_threads ) { if (w.joinable()) w.join(); }
    m_worker_threads.clear();
  }

  void
  ThreadPool3::resize( unsigned thread_count, unsigned queue_capacity ) {
    join();
    if ( queue_capacity == 0 ) queue_capacity = 4 * (thread_count+1);
    m_work_queue.resize( queue_capacity );
    create_workers( thread_count );
  }

}

#endif

//
// eof: ThreadPool3.cc
//
