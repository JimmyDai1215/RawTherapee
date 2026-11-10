/*
 *  This file is part of RawTherapee.
 *
 *  RawTherapee is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  RawTherapee is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with RawTherapee.  If not, see <https://www.gnu.org/licenses/>.
 */
#pragma once

/**
 * Writes a report to "RawTherapee-crash-log.txt" in the user's Downloads
 * folder when the program dies unexpectedly (Windows only): the kind of
 * failure, where it happened, memory figures and third-party modules
 * loaded into the process.
 */
namespace CrashLog
{

void install(const char* version);
/// Call right before a regular shutdown so it is not reported as a crash.
void markNormalExit();

}
