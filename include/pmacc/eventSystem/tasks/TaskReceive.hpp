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

#include "pmacc/Environment.hpp"
#include "pmacc/eventSystem/Manager.hpp"
#include "pmacc/eventSystem/events/EventDataReceive.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class Exchange;

    template<class TYPE, unsigned DIM>
    class TaskReceive : public MPITask
    {
    public:
        TaskReceive(Exchange<TYPE, DIM>& ex) : exchange(&ex), state(Constructor)
        {
        }

        void init() override
        {
            state = WaitForReceived;
            if(Environment<>::get().isMpiDirectEnabled())
            {
                /* Wait to be sure that all device work is finished before MPI is triggered.
                 * MPI will not wait for work in our device streams
                 */
                eventSystem::getTransactionEvent().waitForFinished();
            }
            Environment<>::get().Factory().createTaskReceiveMPI(exchange, this);
        }

        bool executeIntern() override
        {
            switch(state)
            {
            case WaitForReceived:
                break;
            case RunCopy:
                state = WaitForFinish;
                eventSystem::startTransaction();

                /* If MPI direct is enabled
                 *   - we do not have any host representation of an exchange
                 *   - MPI will write directly into the device buffer
                 *     or double buffer when available.
                 */
                if(exchange->hasDeviceDoubleBuffer())
                {
                    if(Environment<>::get().isMpiDirectEnabled())
                    {
                        exchange->getDeviceDoubleBuffer().setSize(newBufferSize);
                    }
                    else
                    {
                        exchange->getHostBuffer().setSize(newBufferSize);
                        Environment<>::get().Factory().createTaskCopy(
                            exchange->getHostBuffer(),
                            exchange->getDeviceDoubleBuffer());
                    }

                    Environment<>::get().Factory().createTaskCopy(
                        exchange->getDeviceDoubleBuffer(),
                        exchange->getDeviceBuffer(),
                        this);
                }
                else
                {
                    if(Environment<>::get().isMpiDirectEnabled())
                    {
                        exchange->getDeviceBuffer().setSize(newBufferSize);
                        /* We can not be notified from setSize() therefore
                         * we need to wait that the current event is finished.
                         */
                        setSizeEvent = eventSystem::getTransactionEvent();
                        state = WaitForSetSize;
                    }
                    else
                    {
                        exchange->getHostBuffer().setSize(newBufferSize);
                        Environment<>::get().Factory().createTaskCopy(
                            exchange->getHostBuffer(),
                            exchange->getDeviceBuffer(),
                            this);
                    }
                }

                eventSystem::endTransaction();
                break;
            case WaitForSetSize:
                // this code is only passed if gpu direct is enabled
                if(nullptr == Manager::getInstance().getITaskIfNotFinished(setSizeEvent.getTaskId()))
                {
                    state = Finish;
                    return true;
                }
                break;
            case WaitForFinish:
                break;
            case Finish:
                return true;
            default:
                return false;
            }

            return false;
        }

        ~TaskReceive() override
        {
            notify(this->myId, RECVFINISHED, nullptr);
        }

        void event(id_t, EventType type, IEventData* data) override
        {
            switch(type)
            {
            case RECVFINISHED:
                if(data != nullptr)
                {
                    auto* rdata = static_cast<EventDataReceive*>(data);
                    // std::cout<<" data rec "<<rdata->getReceivedCount()/sizeof(TYPE)<<std::endl;
                    newBufferSize = rdata->getReceivedCount() / sizeof(TYPE);
                    state = RunCopy;
                    executeIntern();
                }
                break;
            case COPY:
                state = Finish;
                break;
            default:
                return;
            }
        }

        std::string toString() override
        {
            std::stringstream ss;
            ss << state;
            return std::string("TaskReceive ") + ss.str();
        }

    private:
        enum state_t
        {
            Constructor,
            WaitForReceived,
            RunCopy,
            WaitForSetSize,
            WaitForFinish,
            Finish
        };


        Exchange<TYPE, DIM>* exchange;
        state_t state;
        size_t newBufferSize;
        EventTask setSizeEvent;
    };

} // namespace pmacc
