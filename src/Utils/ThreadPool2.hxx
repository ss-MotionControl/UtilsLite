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
// file: ThreadPool2.hxx
//

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
  //! \brief Manages a pool of worker threads for concurrent task execution.
  //!
  //! The `ThreadPool2` class is derived from `ThreadPoolBase` and provides
  //! functionality to execute tasks concurrently using a pool of worker threads.
  //!
  class ThreadPool2 : public ThreadPoolBase {

    using TYPE = std::function<void()>;

    class ThreadsafeQueue {
    private:
      std::queue<TYPE>        m_queue;
      std::mutex              m_mutex;
      std::condition_variable m_cond;
    public:
 
      void
      pop( TYPE & item ) {
        std::unique_lock<std::mutex> mlock(m_mutex);
        while ( m_queue.empty() ) m_cond.wait(mlock);
        item = std::move(m_queue.front());
        m_queue.pop();
      }
 
      void
      push( TYPE const & item ) {
        { std::lock_guard<std::mutex> mlock(m_mutex); m_queue.push(std::move(item)); }
        m_cond.notify_one();
      }
 
      //void
      //push( TYPE && item ) {
      // { std::lock_guard<std::mutex> mlock(m_mutex); m_queue.push(std::move(item)); }
      //  m_cond.notify_one();
      //}

      bool not_empty() const { return !m_queue.empty(); }
    };
      
    std::mutex               m_mutex;
    std::condition_variable  m_cv;
    std::vector<std::thread> m_workers;           //!< fixed number of threads
    ThreadsafeQueue          m_tasks;             //!< submitted tasks
    std::atomic<unsigned>    m_active{0};         //!< number of threads that are executing a task
    bool                     m_is_joining{false}; //!< set to true if join() is invoked

  public:

    //!
    //! \brief Constructs a new ThreadPool1 with a specified number of threads.
    //!
    //! \param nthread Number of threads to create (default: hardware concurrency - 1).
    //!
    explicit
    ThreadPool2(
      unsigned nthread = std::max(
        unsigned(1),
        unsigned(std::thread::hardware_concurrency()-1)
      )
    );

    //!
    //! \brief Destroys the ThreadPool1 and stops all worker threads.
    //!
    virtual
    ~ThreadPool2() {
      join();
    }

    //!
    //! \brief Executes a task in the thread pool.
    //!
    //! \param fun The function to be executed.
    //!
    void exec( TYPE && task ) override { m_tasks.push(std::move(task)); }
    void wait() override; //!< Waits for all tasks to finish
    void join() override; //!< Stops and joins all threads

    //!
    //! \brief Returns the number of threads in the pool.
    //!
    //! \return The number of threads.
    //!
    unsigned
    thread_count() const override
    { return unsigned(m_workers.size()); }

    void resize( unsigned numThreads ) override; //!< Resizes the thread pool
    char const * name() const override { return "ThreadPool2"; } //!< Returns the name of the thread pool

    //!
    //! \brief Returns the ID of the specified worker thread.
    //!
    //! \param i Index of the worker.
    //! \return The thread ID.
    //!
    std::thread::id
    get_id( unsigned i ) const
    { return m_workers[size_t(i)].get_id(); }

    //!
    //! \brief Returns the thread object of the specified worker.
    //!
    //! \param i Index of the worker.
    //! \return The thread object.
    //!
    std::thread const & get_thread( unsigned i ) const { return m_workers[size_t(i)]; }
    std::thread       & get_thread( unsigned i )       { return m_workers[size_t(i)]; }
  };

  /*! @} */

}

//
// eof: ThreadPool2.hxx
//
