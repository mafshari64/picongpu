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
#include "pmacc/communication/ICommunicator.hpp"
#include "pmacc/eventSystem/tasks/MPITask.hpp"

#include <mpi.h>

namespace pmacc
{
    template<class TYPE, unsigned DIM>
    class Exchange;

    template<class TYPE, unsigned DIM>
    class TaskSendMPI : public MPITask
    {
    public:
        TaskSendMPI(Exchange<TYPE, DIM>* exchange) : MPITask(), exchange(exchange)
        {
        }

        void init() override
        {
            auto cPtr = exchange->getCPtrCurrentSize();

            this->request = Environment<DIM>::get().EnvironmentController().getCommunicator().startSend(
                exchange->getExchangeType(),
                cPtr.asCharPtr(),
                cPtr.sizeInBytes(),
                exchange->getCommunicationTag());
        }

        bool executeIntern() override
        {
            if(this->isFinished())
                return true;

            if(this->request == nullptr)
                throw std::runtime_error("request was nullptr (call executeIntern after freed");

            int flag = 0;
            MPI_CHECK(MPI_Test(this->request, &flag, &(this->status)));

            if(flag) // finished
            {
                delete this->request;
                this->request = nullptr;
                this->setFinished();
                return true;
            }
            return false;
        }

        ~TaskSendMPI() override
        {
            notify(this->myId, SENDFINISHED, nullptr);
        }

        void event(id_t, EventType, IEventData*) override
        {
        }

        std::string toString() override
        {
            return std::string("TaskSendMPI exchange type=") + std::to_string(exchange->getExchangeType());
        }

    private:
        Exchange<TYPE, DIM>* exchange;
        MPI_Request* request;
        MPI_Status status;
    };

} // namespace pmacc
