/* -------------------------------------------------------------

This file is a component of SDPA
Copyright (C) 2004 SDPA Project

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA

------------------------------------------------------------- */

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-03: fatal errors exit non-zero. See git log. */
/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-05: Time:: uses a monotonic steady_clock. See git log. */
/*--------------------------------------------------
  rsdpa_tool.h
  $Id: rsdpa_tool.h,v 1.2 2004/09/01 06:34:12 makoto Exp $
--------------------------------------------------*/

#ifndef __sdpa_tool_h__
#define __sdpa_tool_h__

#include <sdpa_right.h>

#include <iostream>
#include <string>
#include <cstdlib>

#include <chrono>

#include <gmpxx.h>

namespace sdpa {

// clang-format off
#define rMessage(message) \
    cout << message << " :: line " << __LINE__ \
         << " in " << __FILE__ << endl

/* MODIFIED from upstream (GPLv2 2a notice), 2026-08-13: rError wrapped in do/while(0).
   As two bare statements it silently broke every unbraced conditional that used it --

       if (cond)
           rError("...");        // expands to:  cout << ...;  exit(EXIT_FAILURE);

   puts only the message under the condition and calls exit() UNCONDITIONALLY. That is not
   hypothetical: written exactly that way, one new call site in sdpa_linear.cpp killed every
   run that enabled the factor digest, instantly and with no output, because the cout never
   ran. It took a debugger to find, since there was no message and no stack -- just exit 1.
   With hundreds of call sites the trap was reloaded for anyone who wrote an ordinary `if`.
   do/while(0) makes the macro one statement, so unbraced conditionals behave as they read.
   See git log. */
#define rError(message)                            \
    do {                                           \
        cout << message << " :: line " << __LINE__ \
             << " in " << __FILE__ << endl;        \
        exit(EXIT_FAILURE);                        \
    } while (0)

#define rNewCheck() ;

// Elapsed wall time (steady_clock, via rGetUseTime).
#define TimeStart(START__) \
   static double START__; START__ = Time::rGetUseTime()
#define TimeEnd(END__) \
   static double END__;   END__ = Time::rGetUseTime()
#define TimeCal(START__,END__) (END__ - START__)
// clang-format on

#define REVERSE_PRIMAL_DUAL 1

// These are constant. Do NOT change
extern int IZERO; // =  0;
extern int IONE;  // =  1;
extern int IMONE; // = -1;

class Time {
  public:
    // Elapsed wall time, NOT process CPU time. times().tms_utime -- what upstream used --
    // sums the CPU of every worker thread, so it grows with the thread count even when the
    // run gets faster, and a real parallel speedup reads as a regression.
    //
    // 2026-08-05: system_clock -> steady_clock. system_clock is NOT monotonic: NTP slew or a
    // step adjustment during a run is added to or subtracted from whatever interval spans it,
    // and a backward step can make a phase report a negative time. Every use of this function
    // is a difference of two calls (see the TimeStart/TimeEnd/TimeCal macros above), so a
    // monotonic clock is what it wanted all along and the epoch is irrelevant. steady_clock is
    // what the qd fork has used since its port. This changes reported timings only -- no value
    // computed from it enters the solution, and regress.sh's hash excludes every timing line --
    // but for a fork whose deliverable IS a benchmark matrix, the clock is part of the product.
    static double rGetUseTime() {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        return std::chrono::duration<double>(now).count();
    }

};

} // namespace sdpa

#endif // __sdpa_tool_h__
