# Private implementation modules

This directory contains the implementation modules used by the public
`methods.cosydelay` package.  The subdirectories are organized by runtime
responsibility (data protocol, candidate generation, fitting, physics checks,
regeneration, and selection); they are not separate methods, baselines, or
historical releases.

Users should import and run only `methods.cosydelay`.  The internal layout is
kept stable so the public runner and its tests can be reproduced without the
research workspace.
