/***************************************************************************
*   � 2012,2014 Advanced Micro Devices, Inc. All rights reserved.
*
*   Licensed under the Apache License, Version 2.0 (the "License");
*   you may not use this file except in compliance with the License.
*   You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
*   Unless required by applicable law or agreed to in writing, software
*   distributed under the License is distributed on an "AS IS" BASIS,
*   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*   See the License for the specific language governing permissions and
*   limitations under the License.

***************************************************************************/
#pragma once
#if !defined( BOLT_AMP_COUNTING_ITERATOR_H )
#define BOLT_AMP_COUNTING_ITERATOR_H
#include "bolt/amp/bolt.h"
#include "bolt/amp/iterator/iterator_traits.h"

/*! \file bolt/amp/iterator/counting_iterator.h
    \brief Return Same Value or counting Value on dereferencing.
*/


namespace bolt {
namespace amp {

    struct counting_iterator_tag
        : public fancy_iterator_tag
        {   // identifying tag for random-access iterators
        };

        /*! \addtogroup fancy_iterators
         */

        /*! \addtogroup AMP-CountingIterator
        *   \ingroup fancy_iterators
        *   \{
        */

        /*! counting_iterator iterates a range with sequential values.
         *
         *
         *
         *  \details The following example demonstrates how to use a \p counting_iterator.
         *
         *  \code
         *  #include <bolt/amp/counting_iterator.h>
         *  #include <bolt/amp/transform.h>
         *  ...
         *
         *  std::vector<int> vecSrc( 5 );
         *  std::vector<int> vecDest( 5 );
         *
         *  std::fill( vecSrc.begin( ), vecSrc.end( ), 10 );
         *
         *  bolt::amp::control ctrl = control::getDefault( );
         *  ...
         *  bolt::amp::counting_iterator< int > count5( 5 );
         *  bolt::amp::transform( ctrl, vecSrc.begin( ), vecSrc.end( ), count5, vecDest.begin( ), bolt::amp::plus< int >( ) );
         *
         *  // Output:
         *  // vecDest = { 15, 16, 17, 18, 19 }
         *
         *  // counting_iterator can save bandwidth when used instead of a range of values.
         *  \endcode
         *
         */

        template< typename value_type >
        class counting_iterator: public std::iterator< counting_iterator_tag, value_type, int>
        {
        public:
             typedef typename std::iterator< counting_iterator_tag, value_type, int>::difference_type
             difference_type;

             typedef concurrency::array_view< value_type > arrayview_type;
             typedef counting_iterator<value_type> const_iterator;


            //  Basic constructor requires a reference to the container and a positional element
	    explicit
            counting_iterator( value_type init, const control& ctl = control::getDefault( ) )
                : m_Index( 0 ), m_initValue( init )  {}

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
           template< typename OtherType >
           counting_iterator( const counting_iterator< OtherType >& rhs ) restrict(cpu, amp)
               : m_Index( rhs.m_Index ), m_initValue( rhs.m_initValue ) {}

            //  This copy constructor allows an iterator to convert into a const_iterator, but not vica versa
            counting_iterator< value_type >& operator= ( const counting_iterator< value_type >& rhs ) restrict(cpu, amp)
            {
                if( this == &rhs )
                    return *this;

                m_initValue = rhs.m_initValue;
                m_Index = rhs.m_Index;
                return *this;
            }

            counting_iterator< value_type >& operator+= ( const  difference_type & n ) restrict(cpu, amp)
            {
                advance( n );
                return *this;
            }

            const counting_iterator< value_type > operator+ ( const difference_type & n ) const restrict(cpu, amp)
            {
                counting_iterator< value_type > result( *this );
                result.advance( n );
                return result;
            }

			const counting_iterator< value_type > operator- ( const difference_type & n ) const restrict(cpu, amp)
            {
                counting_iterator< value_type > result( *this );
                result.advance( -n );
                return result;
            }

            const counting_iterator< value_type > & getBuffer( const_iterator itr ) const restrict(cpu, amp)
            {
                return *this;
            }


			value_type* getPointer() restrict(cpu, amp)
            {
                return &m_initValue;
            }

            const value_type* getPointer() const restrict(cpu, amp)
            {
                return &m_initValue;
            }

            const counting_iterator< value_type > & getContainer( ) const restrict(cpu, amp)
            {
                return *this;
            }

            difference_type operator- ( const counting_iterator< value_type >& rhs ) const restrict(cpu, amp)
            {
                return m_Index - rhs.m_Index;
            }

            //  Public member variables
            difference_type m_Index;

            //  Used for templatized copy constructor and the templatized equal operator
            template < typename > friend class counting_iterator;

            //  For a counting_iterator, do nothing on an advance
            void advance( difference_type n ) restrict(cpu, amp)
            {
                m_Index += n;
            }

            // Pre-increment
            counting_iterator< value_type > operator++ ( ) restrict(cpu, amp)
            {
                advance( 1 );
                //counting_iterator< value_type > result( *this );
                return *this;
            }

            // Post-increment
            counting_iterator< value_type > operator++ ( int ) restrict(cpu, amp)
            {
                advance( 1 );
                return *this;
            }

            // Pre-decrement
            counting_iterator< value_type > operator--( ) restrict(cpu, amp)
            {
                advance( -1 );
                return *this;
            }

            // Post-decrement
            counting_iterator< value_type > operator--( int ) restrict(cpu, amp)
            {
                advance( -1 );
                return *this;
            }

            difference_type getIndex() const restrict(cpu, amp)
            {
                return m_Index;
            }

            template< typename OtherType >
            bool operator== ( const counting_iterator< OtherType >& rhs ) const restrict(cpu, amp)
            {
                bool sameIndex = (rhs.m_initValue == m_initValue) && (rhs.m_Index == m_Index);

                return sameIndex;
            }

            template< typename OtherType >
            bool operator!= ( const counting_iterator< OtherType >& rhs ) const restrict(cpu, amp)
            {
                bool sameIndex = (rhs.m_initValue != m_initValue) || (rhs.m_Index != m_Index);

                return sameIndex;
            }

            template< typename OtherType >
            bool operator< ( const counting_iterator< OtherType >& rhs ) const restrict(cpu, amp)
            {
                bool sameIndex = (m_Index < rhs.m_Index);

                return sameIndex;
            }

            value_type operator*() const restrict(cpu,amp)
            {
                value_type xy = m_initValue + m_Index;
                return xy;
            }


            value_type operator[](int x) const restrict(cpu,amp)
            {
              value_type temp = m_initValue + x;
              return temp;
            }

          private:
            value_type m_initValue;
        };


    template< typename Type >
    static
    inline
    counting_iterator< Type > make_counting_iterator( Type initValue ) restrict(cpu, amp)
    {
        counting_iterator< Type > tmp( initValue );
        return tmp;
    }

}
}


#endif
