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
// file: ThreadPool2.cc
//

#ifndef DOXYGEN_SHOULD_SKIP_THIS

#include "Utils.hh"
#include "Utils_fmt.hh"

namespace Utils {

  ThreadPool2::ThreadPool2( unsigned thread_count )
  : m_workers(thread_count) {
    for ( auto & t : m_workers ) {
      t = std::thread([this]() mutable -> void {
        std::function<void()> task;
        while (!this->m_is_joining) {
          this->m_tasks.pop(task);
          ++this->m_active;
          task();
          --this->m_active;
          this->m_cv.notify_one();
        }
      });
    }
  }

  void
  ThreadPool2::wait() {
    std::unique_lock<std::mutex> lock(m_mutex);
    while ( m_active > 0 || m_tasks.not_empty() ) m_cv.wait(lock);
  }

  void
  ThreadPool2::join() {
    std::unique_lock<std::mutex> lock(m_mutex);
    if ( m_is_joining ) return;
    m_is_joining = true;
    for ( auto & t : m_workers ) m_tasks.push([](){});
    for ( auto & t : m_workers ) if ( t.joinable() ) t.join();
    m_workers.clear();
    m_is_joining = false;
  }

  void
  ThreadPool2::resize( unsigned thread_count ) {
    join();
    new (this) ThreadPool2(thread_count);
  }

}

#endif

//
// eof: ThreadPool2.cc
//
