/* Copyright 2023-2024 Tapish Narwal
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <pmacc/attribute/FunctionSpecifier.hpp>
#include <pmacc/memory/STLTuple.hpp>

#include <tuple>
#include <utility>

namespace picongpu
{
    namespace plugins::binning
    {
        namespace detail
        {
            template<typename TFunc, typename TPmaccTuple, std::size_t... Is>
            HDINLINE constexpr decltype(auto) applyImpl(TFunc&& f, TPmaccTuple&& t, std::index_sequence<Is...>)
            {
                return std::forward<TFunc>(f)(pmacc::memory::tuple::get<Is>(std::forward<TPmaccTuple>(t))...);
            }

            template<typename TFunc, typename TPmaccTuple, std::size_t... Is>
            HDINLINE constexpr decltype(auto) applyEnumerateImpl(
                TFunc&& f,
                TPmaccTuple&& t,
                std::index_sequence<Is...>)
            {
                return std::forward<TFunc>(f)(std::make_pair(
                    std::integral_constant<std::size_t, Is>{},
                    pmacc::memory::tuple::get<Is>(std::forward<TPmaccTuple>(t)))...);
            }

        } // namespace detail

        // takes pmacc::memory::tuple::Tuple
        template<typename TFunc, typename TPmaccTuple>
        HDINLINE constexpr decltype(auto) apply(TFunc&& f, TPmaccTuple&& t)
        {
            return detail::applyImpl(
                std::forward<TFunc>(f),
                std::forward<TPmaccTuple>(t),
                std::make_index_sequence<pmacc::memory::tuple::tuple_size_v<TPmaccTuple>>{});
        }

        // takes pmacc::memory::tuple::Tuple
        template<typename TFunc, typename TPmaccTuple>
        HDINLINE constexpr decltype(auto) applyEnumerate(TFunc&& f, TPmaccTuple&& t)
        {
            return detail::applyEnumerateImpl(
                std::forward<TFunc>(f),
                std::forward<TPmaccTuple>(t),
                std::make_index_sequence<pmacc::memory::tuple::tuple_size_v<TPmaccTuple>>{});
        }

        namespace detail
        {
            template<size_t... Is, typename TPmaccTuple, typename Functor>
            constexpr auto tupleMapHelper(std::index_sequence<Is...>, TPmaccTuple&& tuple, Functor&& functor) noexcept
            {
                return pmacc::memory::tuple::make_tuple(std::forward<Functor>(functor)(
                    pmacc::memory::tuple::get<Is>(std::forward<TPmaccTuple>(tuple)))...);
            }
        } // namespace detail

        /**
         * @brief create a new tuple from the return value of a functor applied on all arguments of a tuple
         */
        template<typename TPmaccTuple, typename Functor>
        constexpr auto tupleMap(TPmaccTuple&& tuple, Functor&& functor) noexcept
        {
            return detail::tupleMapHelper(
                std::make_index_sequence<pmacc::memory::tuple::tuple_size_v<TPmaccTuple>>{},
                std::forward<TPmaccTuple>(tuple),
                std::forward<Functor>(functor));
        }

        template<typename... Args>
        constexpr auto createTuple(Args&&... args) noexcept
        {
            return pmacc::memory::tuple::make_tuple(std::forward<Args>(args)...);
        }

    } // namespace plugins::binning
} // namespace picongpu
