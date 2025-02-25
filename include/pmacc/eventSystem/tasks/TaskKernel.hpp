/* Copyright 2013-2024 Felix Schmitt, Rene Widera, Benjamin Worpitz,
 *                     Alexander Grund
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

#include "pmacc/eventSystem/tasks/DeviceTask.hpp"

namespace pmacc
{
    class TaskKernel : public DeviceTask
    {
    public:
        TaskKernel(std::string kernelName) : DeviceTask(), canBeChecked(false), kernelName(kernelName)
        {
        }

        ~TaskKernel() override
        {
            notify(this->myId, KERNEL, nullptr);
        }

        bool executeIntern() override
        {
            if(canBeChecked)
            {
                return isFinished();
            }
            return false;
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        void activateChecks();

        std::string toString() override
        {
            return std::string("TaskKernel ") + kernelName;
        }

        void init() override
        {
        }

    private:
        bool canBeChecked;
        std::string kernelName;
    };

} // namespace pmacc
