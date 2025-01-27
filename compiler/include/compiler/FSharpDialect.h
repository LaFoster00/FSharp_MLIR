//
// Created by lasse on 1/9/25.
//
#pragma once

#include "compiler/PrecompileHeaders.h"

#include "compiler/FSharpInterfacesDefs.h.inc"

/// Include the auto-generated header file containing the declaration of the fsharp
/// dialect.
#include "compiler/FSharpDialect.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// fsharp operations.
#define GET_OP_CLASSES
#include "compiler/FSharp.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// fsharp types.
#define GET_TYPEDEF_CLASSES
#include "compiler/FSharpTypes.h.inc"

#define GET_ATTRDEF_CLASSES
#include "compiler/FSharpAttrDefs.h.inc"
