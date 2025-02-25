/* Copyright 2013-2024 Felix Schmitt, Heiko Burau, Rene Widera,
 *                     Wolfgang Hoenig, Benjamin Worpitz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "pmacc/attribute/FunctionSpecifier.hpp"
#include "pmacc/dimensions/DataSpace.hpp"

namespace pmacc
{
    template<typename T_Base>
    struct DataBox : T_Base
    {
        using Base = T_Base;
        using typename Base::RefValueType;
        using typename Base::ValueType;

        DataBox() = default;

        HDINLINE DataBox(Base base) : Base{std::move(base)}
        {
        }

        DataBox(DataBox const&) = default;

        HDINLINE decltype(auto) operator()(DataSpace<Base::Dim> const& idx = {}) const
        {
            return Base::operator[](idx);
        }

        HDINLINE decltype(auto) operator()(DataSpace<Base::Dim> const& idx = {})
        {
            return Base::operator[](idx);
        }

        HDINLINE DataBox shift(DataSpace<Base::Dim> const& offset) const
        {
            DataBox result(*this);
            result.m_ptr = const_cast<typename Base::ValueType*>(&((*this)(offset)));
            return result;
        }
    };
} // namespace pmacc
