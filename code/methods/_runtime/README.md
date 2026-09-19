# Runtime support

These modules implement the data preparation, symbolic parsing and validation,
traffic-physics checks, parameter optimization, population evolution, and LLM
interface used by `methods.cosydelay`.

They are an internal support layer. Import the public method through
`methods.cosydelay` rather than depending on this directory's internal module
layout.
