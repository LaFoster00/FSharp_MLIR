//
// Created by lasse on 1/14/25.
//

#include "compiler/Compiler.h"

#include "Grammar.h"
#include "ast/ASTNode.h"
#include "compiler/FSharpDialect.h"
#include "compiler/ASTToMLIR.h"
#include "compiler/FSharpPasses.h"

namespace fsharp::compiler
{
    int FSharpCompiler::compileProgram(InputType inputType, std::string_view inputFilename, Action emitAction,
                                       bool runOptimizations, std::optional<std::string> executableOutputPath)
    {
        FSharpCompiler::inputType = inputType;
        FSharpCompiler::inputFilename = inputFilename;
        FSharpCompiler::emitAction = emitAction;
        FSharpCompiler::runOptimizations = runOptimizations;

        // Register any command line options.
        mlir::registerAsmPrinterCLOptions();
        mlir::registerMLIRContextCLOptions();
        mlir::registerPassManagerCLOptions();

        switch (emitAction)
        {
        case Action::None:
            fmt::print("No action specified (parsing only?), use -emit=<action>\n");
            break;
        case Action::DumpST:
            fmt::print("Syntax tree result:\n");
            break;
        case Action::DumpAST:
            fmt::print("Abstract syntax tree result:\n");
            break;
        case Action::DumpMLIR:
            fmt::print("Base mlir result:\n");
            break;
        case Action::DumpMLIRTypeInference:
            fmt::print("Type infered mlir result:\n");
            break;
        case Action::DumpMLIRAffine:
            fmt::print("Affine mlir result:\n");
            break;
        case Action::DumpMLIRLLVM:
            fmt::print("LLVM-Dialect mlir result:\n");
            break;
        case Action::DumpLLVMIR:
            fmt::print("LLVM IR result:\n");
            break;
        case Action::RunJIT:
            fmt::print("JIT result:\n");
            break;
        case Action::EmitExecutable:
            break;
        }

        if (emitAction == Action::DumpST)
            return dumpST();

        if (emitAction == Action::DumpAST)
            return dumpAST();

        // If we aren't dumping the AST, then we are compiling with/to MLIR.
        mlir::DialectRegistry registry;
        mlir::func::registerAllExtensions(registry);

        mlir::MLIRContext context(registry);
        // Load our Dialect in this MLIR Context.
        context.getOrLoadDialect<mlir::fsharp::FSharpDialect>();

        mlir::OwningOpRef<mlir::ModuleOp> module;
        if (int error = loadAndProcessMLIR(context, module))
            int a = 10;
        //return error;

        // If we aren't exporting to non-mlir, then we are done.
        bool isOutputingMLIR = emitAction <= Action::DumpMLIRLLVM;
        if (isOutputingMLIR)
        {
            module->dump();
            return 0;
        }

        // Check to see if we are compiling to LLVM IR.
        if (emitAction == Action::DumpLLVMIR)
            return dumpLLVMIR(*module);

        // Otherwise, we must be running the jit.
        if (emitAction == Action::RunJIT)
            return runJit(*module);

        if (emitAction == Action::EmitExecutable)
        {
            if (!executableOutputPath.has_value())
                assert("No output path for executable specified!");
            else
                return emitExecutable(*module, executableOutputPath.value());
        }

        llvm::errs() << "No action specified (parsing only?), use -emit=<action>\n";
        return -1;
    }

    std::unique_ptr<fsharpgrammar::ast::Main> FSharpCompiler::parseInputFile(llvm::StringRef filename)
    {
        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(filename);
        if (std::error_code ec = fileOrErr.getError())
        {
            llvm::errs() << "Could not open input file: " << ec.message() << "\n";
            return nullptr;
        }
        auto buffer = fileOrErr.get()->getBuffer();
        return fsharpgrammar::Grammar::parse(buffer, false, emitAction == Action::DumpST, false);
    }

    int FSharpCompiler::loadMLIR(mlir::MLIRContext& context, mlir::OwningOpRef<mlir::ModuleOp>& module)
    {
        // Handle '.fs' input to the compiler.
        if (inputType != InputType::MLIR &&
            !llvm::StringRef(inputFilename).ends_with(".mlir"))
        {
            auto moduleAST = parseInputFile(inputFilename);
            if (!moduleAST)
                return 6;
            module = fsharpgrammar::compiler::MLIRGen::mlirGen(context, moduleAST, inputFilename);
            return !module ? 1 : 0;
        }

        // Otherwise, the input is '.mlir'.
        llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
            llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
        if (std::error_code ec = fileOrErr.getError())
        {
            llvm::errs() << "Could not open input file: " << ec.message() << "\n";
            return -1;
        }

        // Parse the input mlir.
        llvm::SourceMgr sourceMgr;
        sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
        module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
        if (!module)
        {
            llvm::errs() << "Error can't load file " << inputFilename << "\n";
            return 3;
        }
        return 0;
    }

    int FSharpCompiler::loadAndProcessMLIR(mlir::MLIRContext& context, mlir::OwningOpRef<mlir::ModuleOp>& module)
    {
        if (int error = loadMLIR(context, module))
            return error;

        mlir::PassManager pm(module.get()->getName());
        // Apply any generic pass manager command line options and run the pipeline.
        if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
            return 4;

        // Check to see what granularity of MLIR we are compiling to.
        bool isTypeInference = emitAction >= Action::DumpMLIRTypeInference;
        bool isLoweringToAffine = emitAction >= Action::DumpMLIRAffine;
        bool isLoweringToLLVM = emitAction >= Action::DumpMLIRLLVM;

        if (isTypeInference)
        {
            pm.addPass(mlir::fsharp::createTypeInferencePass());
        }

        if (runOptimizations || isLoweringToAffine)
        {
            // Inline all functions into main and then delete them.
            pm.addPass(mlir::createInlinerPass());

            mlir::OpPassManager& optPM = pm.nest<mlir::func::FuncOp>();

            optPM.addPass(mlir::createCanonicalizerPass());
            optPM.addPass(mlir::createCSEPass());
            optPM.addPass(mlir::bufferization::createEmptyTensorEliminationPass());
        }

        if (isLoweringToAffine)
        {
            // Partially lower the fsharp dialect.
            pm.addPass(mlir::fsharp::createLowerToFunctionPass());
            pm.addPass(mlir::fsharp::createLowerToArithPass());

            // Add a few cleanups post lowering.
            mlir::OpPassManager& optPM = pm.nest<mlir::func::FuncOp>();
            optPM.addPass(mlir::createCanonicalizerPass());
            optPM.addPass(mlir::createCSEPass());

            // Add optimizations if enabled.
            if (runOptimizations)
            {
                optPM.addPass(mlir::affine::createLoopFusionPass());
                optPM.addPass(mlir::affine::createAffineScalarReplacementPass());
            }

            // Bufferize the program
            mlir::bufferization::OneShotBufferizationOptions bufferizationOptions{};
            bufferizationOptions.bufferizeFunctionBoundaries = true;

            pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
            optPM.addPass(mlir::bufferization::createBufferHoistingPass());
            optPM.addPass(mlir::bufferization::createBufferLoopHoistingPass());
            pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());
            pm.addPass(mlir::bufferization::createDropEquivalentBufferResultsPass());
            optPM.addPass(mlir::bufferization::createPromoteBuffersToStackPass());
            optPM.addPass(mlir::bufferization::createBufferDeallocationPass());
        }

        if (isLoweringToLLVM)
        {
            // Finish lowering the fsharp IR to the LLVM dialect.
            pm.addPass(mlir::fsharp::createLowerToLLVMPass());
            pm.addPass(mlir::createReconcileUnrealizedCastsPass());
            // This is necessary to have line tables emitted and basic
            // debugger working. In the future we will add proper debug information
            // emission directly from our frontend.
            pm.addPass(mlir::LLVM::createDIScopeForLLVMFuncOpPass());

        }

        if (mlir::failed(pm.run(*module)))
            return 4;
        return 0;
    }

    int FSharpCompiler::dumpST()
    {
        if (inputType == InputType::MLIR)
        {
            llvm::errs() << "Can't dump a FSharp AST when the input is MLIR\n";
            return 5;
        }

        auto mainAst = parseInputFile(inputFilename);
        if (!mainAst)
            return 1;

        return 0;
    }

    int FSharpCompiler::dumpAST()
    {
        if (inputType == InputType::MLIR)
        {
            llvm::errs() << "Can't dump a FSharp AST when the input is MLIR\n";
            return 5;
        }

        auto mainAst = parseInputFile(inputFilename);
        if (!mainAst)
            return 1;

        std::cout << utils::to_string(*mainAst);
        return 0;
    }

    int FSharpCompiler::dumpLLVMIR(mlir::ModuleOp module)
    {
        // Register the translation to LLVM IR with the MLIR context.
        mlir::registerBuiltinDialectTranslation(*module->getContext());
        mlir::registerLLVMDialectTranslation(*module->getContext());

        // Convert the module to LLVM IR in a new LLVM IR context.
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule)
        {
            llvm::errs() << "Failed to emit LLVM IR\n";
            return -1;
        }

        // Initialize LLVM targets.
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        // Configure the LLVM Module
        auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!tmBuilderOrError)
        {
            llvm::errs() << "Could not create JITTargetMachineBuilder\n";
            return -1;
        }

        auto tmOrError = tmBuilderOrError->createTargetMachine();
        if (!tmOrError)
        {
            llvm::errs() << "Could not create TargetMachine\n";
            return -1;
        }
        mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                              tmOrError.get().get());

        /// Optionally run an optimization pipeline over the llvm module.
        auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/runOptimizations ? 3 : 0, /*sizeLevel=*/0,
                         /*targetMachine=*/nullptr);
        if (auto err = optPipeline(llvmModule.get()))
        {
            llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
            return -1;
        }
        llvm::errs() << *llvmModule << "\n";
        return 0;
    }

    int FSharpCompiler::runJit(mlir::ModuleOp module)
    {
        // Initialize LLVM targets.
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        // Register the translation from MLIR to LLVM IR, which must happen before we
        // can JIT-compile.
        mlir::registerBuiltinDialectTranslation(*module->getContext());
        mlir::registerLLVMDialectTranslation(*module->getContext());

        // An optimization pipeline to use within the execution engine.
        auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/runOptimizations ? 3 : 0, /*sizeLevel=*/0,
                         /*targetMachine=*/nullptr);

        // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
        // the module.
        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.transformer = optPipeline;
        auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
        assert(maybeEngine && "failed to construct an execution engine");
        auto& engine = maybeEngine.get();

        // Invoke the JIT-compiled function.
        auto invocationResult = engine->invokePacked("main");
        if (invocationResult)
        {
            llvm::errs() << "JIT invocation failed\n";
            return -1;
        }

        return 0;
    }

    int FSharpCompiler::emitExecutable(mlir::ModuleOp module, const std::string& outputFilePath,
                                       const bool writeNewDbgInfoFormat)
    {
        // Register the translation to LLVM IR with the MLIR context.
        mlir::registerBuiltinDialectTranslation(*module->getContext());
        mlir::registerLLVMDialectTranslation(*module->getContext());

        // Convert the module to LLVM IR in a new LLVM IR context.
        llvm::LLVMContext llvmContext;
        auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
        if (!llvmModule)
        {
            llvm::errs() << "Failed to emit LLVM IR\n";
            return -1;
        }

        // Initialize LLVM targets.
        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        // Configure the LLVM Module
        auto tmBuilderOrError = llvm::orc::JITTargetMachineBuilder::detectHost();
        if (!tmBuilderOrError)
        {
            llvm::errs() << "Could not create JITTargetMachineBuilder\n";
            return -1;
        }

        auto tmOrError = tmBuilderOrError->createTargetMachine();
        if (!tmOrError)
        {
            llvm::errs() << "Could not create TargetMachine\n";
            return -1;
        }
        mlir::ExecutionEngine::setupTargetTripleAndDataLayout(llvmModule.get(),
                                                              tmOrError.get().get());

        /// Optionally run an optimization pipeline over the llvm module.
        auto optPipeline = mlir::makeOptimizingTransformer(
            /*optLevel=*/runOptimizations ? 3 : 0, /*sizeLevel=*/0,
                         /*targetMachine=*/nullptr);
        if (auto err = optPipeline(llvmModule.get()))
        {
            llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
            return -1;
        }
        std::string llvm_str;
        llvm::raw_string_ostream rso(llvm_str);
        llvmModule->print(rso, nullptr);
        rso.flush();
        std::ofstream out(outputFilePath + ".ll");
        out << rso.str();
        out.close();

        // Link the object file into an executable.
        std::string bytecode_command = "llvm-as-18 " + outputFilePath + ".ll -o " + outputFilePath + ".bc";
        if (system(bytecode_command.c_str()) != 0)
        {
            llvm::errs() << "Failed to emit bytecode\n";
            return -1;
        }

        // Link the object file into an executable.
        std::string compile_command = "clang-18 " + outputFilePath + ".bc -o " + outputFilePath;
        if (system(compile_command.c_str()) != 0)
        {
            llvm::errs() << "Failed to link executable\n";
            return -1;
        }

        llvm::outs() << "Executable emitted to " << outputFilePath << "\n";
        return 0;
    }
}
