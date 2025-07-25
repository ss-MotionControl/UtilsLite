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
 |      Università degli Studi di Trento                                    |
 |      email: enrico.bertolazzi@unitn.it                                   |
 |                                                                          |
\*--------------------------------------------------------------------------*/

//
// file: Malloc.hxx
//

/*\
:|:    ____            _       _             __
:|:   / ___| _     _  (_)_ __ | |_ ___ _ __ / _| __ _  ___ ___
:|:  | |   _| |_ _| |_| | '_ \| __/ _ \ '__| |_ / _` |/ __/ _ \
:|:  | |__|_   _|_   _| | | | | ||  __/ |  |  _| (_| | (_|  __/
:|:   \____||_|   |_| |_|_| |_|\__\___|_|  |_|  \__,_|\___\___|
\*/

namespace Utils {

  /*!
   * \addtogroup Malloc
   * @{
   */

  #ifndef DOXYGEN_SHOULD_SKIP_THIS
  using std::int64_t;
  using std::string;
#ifndef UTILS_MINIMAL_BUILD
  using std::mutex;
#endif
  using std::size_t;
  #endif

#ifndef UTILS_MINIMAL_BUILD
  //! Global mutex for thread-safe memory operations.
  extern std::mutex MallocMutex;
#endif

  //! Global variables for tracking memory allocation statistics.
  extern int64_t CountAlloc;
  extern int64_t CountFreed;
  extern int64_t AllocatedBytes;
  extern int64_t MaximumAllocatedBytes;
  extern bool    MallocDebug;

#ifndef UTILS_MINIMAL_BUILD
  //! Utility function to convert byte size into a human-readable format.
  /*!
   * \param nb The number of bytes.
   * \return A string representing the size in human-readable format (KB, MB, etc.).
   */
  string out_bytes( size_t nb );
#endif

  /*\
  :|:   __  __       _ _
  :|:  |  \/  | __ _| | | ___   ___
  :|:  | |\/| |/ _` | | |/ _ \ / __|
  :|:  | |  | | (_| | | | (_) | (__
  :|:  |_|  |_|\__,_|_|_|\___/ \___|
  \*/

  //! Class for dynamic memory allocation of objects.
  /*!
   * This class provides custom memory management utilities for allocating,
   * freeing, and managing dynamic memory for objects of type `T`.
   *
   * \tparam T The type of objects to allocate.
   */
  template <typename T>
  class Malloc {
  public:
    //! Type alias for the type of objects managed by this allocator.
    using valueType = T;

  private:

    string m_name;                   //!< Name identifier for the allocated memory.
    size_t m_num_total_values{0};    //!< Total number of objects allocated.
    size_t m_num_total_reserved{0};  //!< Total reserved space.
    size_t m_num_allocated{0};       //!< Number of currently allocated objects.
    valueType * m_p_memory{nullptr};      //!< Pointer to the allocated memory.

    //! Internal method to allocate memory for a specified number of objects.
    void allocate_internal( size_t n );

    //! Handle memory exhaustion errors.
    void memory_exausted( size_t sz );

    //! Handle errors when attempting to pop more than allocated.
    void pop_exausted( size_t sz );

  public:

    //! Copy constructor is deleted.
    Malloc( Malloc<T> const & ) = delete;

    //! Assignment operator is deleted.
    Malloc<T> const & operator = ( Malloc<T> const & ) const = delete;

    //! Constructor.
    /*!
     * \param name A string identifier for the allocated memory block.
     */
    explicit
    Malloc( string name )
    : m_name(std::move(name))
    { }

    //! Destructor.
    /*!
     * Frees the allocated memory.
     */
    ~Malloc() { hard_free(); }

    //! Allocate memory for `n` objects, error if already allocated.
    /*!
     * \param n Number of objects to allocate.
     */
    void allocate( size_t n );

    template <typename T2>
    void
    allocate(T2 n) {
      static_assert(std::is_integral<T2>::value, "allocate() accepts only integral types!");
      allocate(static_cast<size_t>(n));
    }

    //! Reallocate memory for `n` objects, even if already allocated.
    /*!
     * \param n Number of objects to reallocate.
     */
    void reallocate( size_t n );

    template <typename T2>
    void
    reallocate(T2 n) {
      static_assert(std::is_integral<T2>::value, "reallocate() accepts only integral types!");
      reallocate(static_cast<size_t>(n));
    }

    //! Free memory without deallocating the pointer.
    void free() { m_num_total_values = m_num_allocated = 0; }

    //! Free memory and deallocate the pointer.
    void hard_free();

    //! Get the number of allocated objects.
    /*!
     * \return Number of currently allocated objects.
     */
    size_t size() const { return m_num_total_values; }

    //! Allocate memory for `sz` objects and return the pointer.
    /*!
     * \param sz Number of objects to allocate.
     * \return Pointer to the allocated memory.
     */
    T * operator () ( size_t sz ) {
      size_t offs = m_num_allocated;
      m_num_allocated += sz;
      if ( m_num_allocated > m_num_total_values ) memory_exausted( sz );
      return m_p_memory + offs;
    }

    template <typename T2>
    T* operator() ( T2 sz ) {
      static_assert(std::is_integral<T2>::value, "operator() accepts only integral types!");
      return (*this)(static_cast<size_t>(sz));
    }
    //! Free memory for `sz` objects.
    /*!
     * \param sz Number of objects to free.
     */
    void
    pop( size_t sz ) {
      if ( sz > m_num_allocated ) pop_exausted( sz );
      m_num_allocated -= sz;
    }

    template <typename T2>
    void
    pop( T2 n ) {
      static_assert(std::is_integral<T2>::value, "pop() accepts only integral types!");
      pop(static_cast<size_t>(n));
    }

    //! Allocate memory for `n` objects.
    /*!
     * \param n Number of objects to allocate.
     * \return Pointer to the allocated memory.
     */
    T * malloc( size_t n );

    template <typename T2>
    T *
    malloc( T2 n ) {
      static_assert(std::is_integral<T2>::value, "malloc() accepts only integral types!");
      return malloc(static_cast<size_t>(n));
    }

    //! Reallocate memory for `n` objects.
    /*!
     * \param n Number of objects to reallocate.
     * \return Pointer to the reallocated memory.
     */
    T * realloc( size_t n );

    template <typename T2>
    T *
    realloc( T2 n ) {
      static_assert(std::is_integral<T2>::value, "realloc() accepts only integral types!");
      return realloc(static_cast<size_t>(n));
    }

    //! Check if the memory is fully allocated.
    /*!
     * \return `true` if all memory is allocated, `false` otherwise.
     */
    bool is_empty() const { return m_num_allocated >= m_num_total_values; }

    //! Ensure that memory is fully used.
    /*!
     * \param where Identifier for where the check is performed.
     */
    void must_be_empty( string_view where ) const;

#ifndef UTILS_MINIMAL_BUILD
    //! Get memory allocation information.
    /*!
     * \param where Identifier for where the information is retrieved.
     * \return A string containing information about memory allocation.
     */
    string info( string_view where ) const;
#endif
  };

  extern template class Malloc<char>;
  extern template class Malloc<uint16_t>;
  extern template class Malloc<int16_t>;
  extern template class Malloc<uint32_t>;
  extern template class Malloc<int32_t>;
  extern template class Malloc<uint64_t>;
  extern template class Malloc<int64_t>;
  extern template class Malloc<float>;
  extern template class Malloc<double>;

  extern template class Malloc<void*>;
  extern template class Malloc<char*>;
  extern template class Malloc<uint16_t*>;
  extern template class Malloc<int16_t*>;
  extern template class Malloc<uint32_t*>;
  extern template class Malloc<int32_t*>;
  extern template class Malloc<uint64_t*>;
  extern template class Malloc<int64_t*>;
  extern template class Malloc<float*>;
  extern template class Malloc<double*>;

  /*\
  :|:   __  __       _ _            _____ _              _
  :|:  |  \/  | __ _| | | ___   ___|  ___(_)_  _____  __| |
  :|:  | |\/| |/ _` | | |/ _ \ / __| |_  | \ \/ / _ \/ _` |
  :|:  | |  | | (_| | | | (_) | (__|  _| | |>  <  __/ (_| |
  :|:  |_|  |_|\__,_|_|_|\___/ \___|_|   |_/_/\_\___|\__,_|
  \*/

  //! Class for fixed-size memory allocation of objects.
  /*!
   * This class manages memory for a fixed number of objects of type `T`.
   *
   * \tparam T The type of objects to allocate.
   * \tparam mem_size The fixed size of memory to allocate.
   */
  template <typename T, std::size_t mem_size>
  class MallocFixed {
  public:
    //! Type alias for the type of objects managed by this allocator.
    using valueType = T;

  private:

    string    m_name;             //!< Name identifier for the allocated memory.
    size_t    m_num_allocated{0}; //!< Number of currently allocated objects.
    valueType m_data[mem_size];   //!< Array to store objects of type `T`.

  public:

    //! Copy constructor is deleted.
    MallocFixed(MallocFixed<T,mem_size> const &) = delete; // blocco costruttore di copia

    //! Assignment operator is deleted.
    MallocFixed<T,mem_size> const & operator = (MallocFixed<T,mem_size> const &) const = delete; // blocco copia

    //! Constructor.
    /*!
     * \param name A string identifier for the allocated memory block.
     */
    explicit
    MallocFixed( string name )
    : m_name(std::move(name))
    {}

    //! Destructor.
    ~MallocFixed() = default;

    //! Free memory without deallocating the pointer.
    void free() { m_num_allocated = 0; }

    //! Get the number of allocated objects.
    /*!
     * \return Number of objects that can be allocated.
     */
    static size_t size() { return mem_size; }

    //! Allocate memory for `sz` objects and return the pointer.
    /*!
     * \param sz Number of objects to allocate.
     * \return Pointer to the allocated memory.
     */
    T * operator () ( size_t sz );

    //! Free memory for `sz` objects.
    /*!
     * \param sz Number of objects to free.
     */
    void pop( size_t sz );

    //! Check if the memory is fully allocated.
    /*!
     * \return `true` if all memory is allocated, `false` otherwise.
     */
    bool is_empty() const { return m_num_allocated >= mem_size; }

  };

  /*! @} */

}

//
// eof: Malloc.hxx
//
