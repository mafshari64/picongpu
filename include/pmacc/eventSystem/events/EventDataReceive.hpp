/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Wolfgang Hoenig,
 *                     Benjamin Worpitz
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

#include "pmacc/eventSystem/events/IEventData.hpp"

namespace pmacc
{
    class EventDataReceive : public IEventData
    {
    public:
        EventDataReceive(EventNotify* task, size_t recv_count) : IEventData(task), recv_count(recv_count)
        {
        }

        size_t getReceivedCount() const
        {
            return recv_count;
        }

    private:
        size_t recv_count;
    };

} // namespace pmacc
