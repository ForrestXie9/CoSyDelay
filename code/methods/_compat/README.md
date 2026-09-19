# Compatibility support

The final CoSyDelay package retains a small compatibility layer because parts
of the numerical implementation use stable absolute import names from the
research code. These files are runtime dependencies of the public package;
they are not separate published methods, variants, or run records.

The public entry point remains `methods.cosydelay`.
