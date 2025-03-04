﻿open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Tokenization

let getSourceTokenizer (file, input) =
    let sourceTok = FSharpSourceTokenizer([], file, None, Some true)
    let tokenizer = sourceTok.CreateLineTokenizer(input)
    tokenizer

/// Tokenize a single line of F# code
let rec tokenizeLine (tokenizer: FSharpLineTokenizer) state =
    match tokenizer.ScanToken(state) with
    | Some tok, state ->
        // Print token name
        printf "%s " tok.TokenName
        // Tokenize the rest, in the new state
        tokenizeLine tokenizer state
    | None, state -> state

/// Print token names for multiple lines of code
let rec tokenizeLines (sourceTok: FSharpSourceTokenizer) state count lines =
    match lines with
    | line :: lines ->
        // Create tokenizer & tokenize single line
        printfn "\nLine %d" count
        let tokenizer = sourceTok.CreateLineTokenizer(line)
        let state = tokenizeLine tokenizer state
        // Tokenize the rest using new state
        tokenizeLines sourceTok state (count + 1) lines
    | [] -> ()

/// Get untyped tree for a specified input
let getUntypedTree (file, input) =
    let checker = FSharpChecker.Create()
    let inputSource = SourceText.ofString input
    // Get compiler options for the 'project' implied by a single script file
    let projOptions, diagnostics =
        checker.GetProjectOptionsFromScript(file, inputSource, assumeDotNetFramework = false)
        |> Async.RunSynchronously

    let parsingOptions, _errors =
        checker.GetParsingOptionsFromProjectOptions(projOptions)

    // Run the first phase (untyped parsing) of the compiler
    let parseFileResults =
        checker.ParseFile(file, inputSource, parsingOptions) |> Async.RunSynchronously

    parseFileResults.ParseTree


let printAst (ast: ParsedInput) =
    match ast with
    | ParsedInput.ImplFile parsedImplFileInput -> printfn "Implementation File: %A" parsedImplFileInput
    | ParsedInput.SigFile parsedSigFileInput -> printfn "Signature File: %A" parsedSigFileInput


let readFileAsString (filePath: string) : string = File.ReadAllText(filePath)

// Example usage
let filePath = "Program.fs"
let fileContents = readFileAsString filePath

let lines = fileContents.Split('\r', '\n')

let sourceTok = FSharpSourceTokenizer([], Some "Test.fs", None, Some true)

lines |> List.ofSeq |> tokenizeLines sourceTok FSharpTokenizerLexState.Initial 1

// Inspect the syntax tree
getUntypedTree (filePath, fileContents) |> printAst


let test x y = 
    if x && y then
        printf "x and y are true"
    else
        printf "x and y are not true"


let add a b = a + b
assert (add 1 2 = 3) "add 1 2 = 3"