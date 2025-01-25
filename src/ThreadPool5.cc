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
// file: ThreadPool5.cc
//

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#if defined(__llvm__) || defined(__clang__)
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wduplicate-enum"
#endif

#include "Utils.hh"
#include "Utils_fmt.hh"

namespace Utils {
  /*\
   |   _____ _                        _ ____             _
   |  |_   _| |__  _ __ ___  __ _  __| |  _ \ ___   ___ | |
   |    | | | '_ \| '__/ _ \/ _` |/ _` | |_) / _ \ / _ \| |
   |    | | | | | | | |  __/ (_| | (_| |  __/ (_) | (_) | |
   |    |_| |_| |_|_|  \___|\__,_|\__,_|_|   \___/ \___/|_|
  \*/

  ThreadPool5::ThreadPool5( unsigned nthread )
  : ThreadPoolBase()
  {
    resize_workers( nthread );
    //info( std::cout );
  }

  ThreadPool5::~ThreadPool5() {
    join();
    m_workers.clear();
    m_stack.clear();
  }

  void
  ThreadPool5::resize_workers( unsigned numThreads ) {
    m_stack.clear(); // empty stack
    m_stack.reserve( size_t(numThreads) );
    m_workers.resize( size_t(numThreads) );
    unsigned id = 0;
    for ( Worker & w : m_workers ) { w.setup( this, id ); ++id; }
    while ( id-- > 0 ) push_worker( id );
    setup();
  }

  void
  ThreadPool5::push_worker( unsigned id ) {
    std::unique_lock<std::mutex> lock(m_stack_mutex);
    m_stack.emplace_back(id);
    m_stack_cond.notify_one();
  }

  unsigned
  ThreadPool5::pop_worker() {
    std::unique_lock<std::mutex> lock(m_stack_mutex);
    m_stack_cond.wait( lock, [&]()->bool { return !m_stack.empty(); } );
    unsigned id = m_stack.back(); m_stack.pop_back();
    return id;
  }

  /*\
   |  __        __         _
   |  \ \      / /__  _ __| | _____ _ __
   |   \ \ /\ / / _ \| '__| |/ / _ \ '__|
   |    \ V  V / (_) | |  |   <  __/ |
   |     \_/\_/ \___/|_|  |_|\_\___|_|
  \*/

  void
  ThreadPool5::Worker::start() {
    if ( !m_active ) {
      m_active = true;
      m_running_thread = std::thread( &Worker::worker_loop, this );
    }
  }

  void
  ThreadPool5::Worker::stop() {
    if ( m_active ) {
      wait();               // if running task wait it terminate
      m_active = false;     // deactivate computation
      m_job = [](){};       // dummy task
      m_is_running.green(); // start computation (exiting loop)
      if ( m_running_thread.joinable() ) m_running_thread.join(); // wait thread for exiting
      m_is_running.red();   // end of computation (for double stop);
    }
    //fmt::print( "worker_loop {} stopped\n", m_worker_id );
  }

  void
  ThreadPool5::Worker::worker_loop() {
    m_is_running.red(); // block computation
    while ( m_active ) {
      m_is_running.wait(); // wait signal to start computation
      // ----------------------------------------
      if ( !m_active ) break; // if finished exit
      m_job();
      // ----------------------------------------
      m_is_running.red();     // block computation
      ++m_job_done_counter;
      m_tp->push_worker( m_worker_id ); // worker ready for a new computation
      std::this_thread::yield();
    }
  }

}

#endif

//
// eof: ThreadPool5.cc
//
