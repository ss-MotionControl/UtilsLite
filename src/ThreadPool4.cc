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
// file: ThreadPool4.cc
//

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#if defined(__llvm__) || defined(__clang__)
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wduplicate-enum"
#endif

#include "Utils.hh"
#include "Utils_fmt.hh"

namespace Utils {

  ThreadPool4::ThreadPool4( unsigned thread_count, unsigned queue_capacity )
  : m_done(false)
  , m_running_task(0)
  , m_running_thread(0)
  , m_work_queue( queue_capacity == 0 ? 4 * (thread_count+1) : queue_capacity )
  , m_pop_waiting(0)
  , m_push_waiting(0)
  {
    create_workers( thread_count );
  }

  void
  ThreadPool4::push_task( TaskData * task ) {
    // --------------------------
    ++m_push_waiting;
    { std::unique_lock<std::mutex> lock( m_work_on_queue_mutex );
      while ( m_work_queue.is_full() ) m_queue_push_cv.wait( m_work_on_queue_mutex );
      m_work_queue.push( task );
      --m_push_waiting;
    }
    if ( m_pop_waiting > 0 ) m_queue_pop_cv.notify_one();
    if ( m_push_waiting > 0 && !m_work_queue.is_full() ) m_queue_push_cv.notify_one();
  }

  tp::Queue::TaskData *
  ThreadPool4::pop_task() {
    TaskData * task{ nullptr };
    ++m_pop_waiting;
    { std::unique_lock<std::mutex> lock(m_work_on_queue_mutex);
      while ( m_work_queue.empty() ) m_queue_pop_cv.wait( m_work_on_queue_mutex );
      task = m_work_queue.pop();
      ++m_running_task; // must be incremented in the locked part
      --m_pop_waiting;
    }
    if ( m_push_waiting > 0 ) m_queue_push_cv.notify_one();
    if ( m_pop_waiting  > 0 && !m_work_queue.empty() ) m_queue_pop_cv.notify_one();
    return task;
  }

  void
  ThreadPool4::worker_thread() {
    ++m_running_thread;
    while ( !m_done ) {
      // ---------------------------- POP
      TaskData * task = pop_task();
      // ---------------------------- RUN
      (*task)(); // run and delete task;
      // ---------------------------- UPDATE
      --m_running_task;
    }
    --m_running_thread;
  }

  void
  ThreadPool4::create_workers( unsigned thread_count ) {
    m_worker_threads.clear();
    m_worker_threads.reserve(thread_count);
    m_done = false;
    try {
      for ( unsigned i=0; i<thread_count; ++i )
        m_worker_threads.emplace_back( &ThreadPool4::worker_thread, this );
    } catch(...) {
      m_done = true;
      throw;
    }
  }

  void
  ThreadPool4::resize( unsigned thread_count, unsigned queue_capacity ) {
    join();
    if ( queue_capacity == 0 ) queue_capacity = 4 * (thread_count+1);
    m_work_queue.resize( queue_capacity );
    create_workers( thread_count );
  }

  void
  ThreadPool4::wait()
  { while ( !m_work_queue.empty() || m_running_task > 0 ) nano_sleep(); }

  void
  ThreadPool4::join() {
    this->wait(); // finish all the running task
    m_done = true;
    unsigned i = m_running_thread;
    while ( i-- > 0 ) push_task( new TaskData([](){}) );
    while ( m_running_thread > 0 ) nano_sleep();
    m_work_queue.clear(); // remove spurious (null task) remained
    for ( std::thread & w : m_worker_threads ) { if (w.joinable()) w.join(); }
    m_worker_threads.clear(); // destroy the workers threads vector
  }

}

#endif

//
// eof: ThreadPool4.cc
//
