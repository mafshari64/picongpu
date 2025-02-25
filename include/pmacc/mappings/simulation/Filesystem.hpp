/* Copyright 2014-2024 Felix Schmitt
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

#include "pmacc/Environment.def"
#include "pmacc/types.hpp"

#include <string>

namespace pmacc
{
    /**
     * Singleton class providing common filesystem operations.
     */
    class Filesystem
    {
    public:
        /**
         * Create directory with default permissions
         *
         * @param dir name of directory
         */
        void createDirectory(const std::string dir) const;
        /**
         * Set 755 permissions for a directory
         *
         * @param dir name of directory
         */
        void setDirectoryPermissions(const std::string dir) const;

        /**
         * Create directory and set 755 permissions by root rank.
         *
         * @param dir name of directory
         */
        void createDirectoryWithPermissions(const std::string dir) const;

        /**
         * Strip path from absolute or relative paths to filenames
         *
         * @param path and filename
         */
        std::string basename(const std::string pathFilename) const;

        /**
         * Returns the instance of the filesystem class.
         *
         * This class is a singleton class.
         *
         * @return a filesystem instance
         */
        static Filesystem& get()
        {
            static Filesystem instance;
            return instance;
        }

    private:
        /**
         * Constructor
         */
        Filesystem() = default;

        /**
         * Constructor
         */
        Filesystem(const Filesystem& fs) = default;
    };

} // namespace pmacc
