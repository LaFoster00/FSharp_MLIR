open System.IO
open FSharp.Compiler.CodeAnalysis
open FSharp.Compiler.Text
open FSharp.Compiler.Syntax
open FSharp.Compiler.Tokenization

let a: int = 10 * 10
let b =
    let c =
        10 + 10
    c + 10
let c: float = float(10 + 20 * 30 - 50)

let d = if 10 > 20 then 30 else 40
let e =
    if 10 > 20 then
        30
    else
        40

let f x y z = x + y + z
let g = f 10 e 30


let getSourceTokenizer (file, input) =
    let sourceTok = FSharpSourceTokenizer([], file, None, Some true)
    let tokenizer = sourceTok.CreateLineTokenizer(input)
    tokenizer

 /// Tokenize a single line of F# code
let rec tokenizeLine (tokenizer:FSharpLineTokenizer) state =
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
    | line::lines ->
        // Create tokenizer & tokenize single line
        printfn "\nLine %d" count
        let tokenizer = sourceTok.CreateLineTokenizer(line)
        let state = tokenizeLine tokenizer state
        // Tokenize the rest using new state
        tokenizeLines sourceTok state (count+1) lines
    | [] -> ()

/// Get untyped tree for a specified input
let getUntypedTree (file, input) =
    let checker = FSharpChecker.Create()
    let inputSource = SourceText.ofString input
    // Get compiler options for the 'project' implied by a single script file
    let projOptions, diagnostics = 
        checker.GetProjectOptionsFromScript(file, inputSource, assumeDotNetFramework=false)
        |> Async.RunSynchronously

    let parsingOptions, _errors = checker.GetParsingOptionsFromProjectOptions(projOptions)

    // Run the first phase (untyped parsing) of the compiler
    let parseFileResults = 
        checker.ParseFile(file, inputSource, parsingOptions) 
        |> Async.RunSynchronously

    parseFileResults.ParseTree

    
let printAst (ast: ParsedInput) =
    match ast with
    | ParsedInput.ImplFile parsedImplFileInput ->
        printfn "Implementation File: %A" parsedImplFileInput
    | ParsedInput.SigFile parsedSigFileInput ->
        printfn "Signature File: %A" parsedSigFileInput


let readFileAsString (filePath: string) : string =
    File.ReadAllText(filePath)

// Example usage
let filePath = "Program.fs"
let fileContents = readFileAsString filePath

let lines = fileContents.Split('\r', '\n')

let sourceTok = FSharpSourceTokenizer([], Some "Test.fs", None, Some true)

lines
|> List.ofSeq
|> tokenizeLines sourceTok FSharpTokenizerLexState.Initial 1

// Inspect the syntax tree
getUntypedTree (filePath, fileContents)
|> printAst

//Pattern matching
let tuple_pat = (1, "hello", true)
match tuple_pat with
| (1, "hello", true) -> printfn "Matched tuple (1, \"hello\", true)"
| _ -> printfn "No match"

let and_pat = (1, "hello", 3)
match and_pat with
| (1, "hello", 3) & (1, _, _) -> printfn "Matched first part and second part"
| _ -> printfn "No match"

let or_pat = "apple"
match or_pat with
| "apple" | "banana" -> printfn "Matched apple or banana"
| _ -> printfn "No match"

let as_pat = Some(10)
match as_pat with
| Some x as value -> printfn "Matched Some with value: %d, full match: %A" x value
| None -> printfn "Matched None"

let cons_pat = [1; 2; 3]
match cons_pat with
| 1 :: tail -> printfn "Matched 1 as head, and tail is: %A" tail
| _ -> printfn "No match"

let typed_pat = "hello"
match typed_pat with
| :? string as str -> printfn "Matched a string: %s" str
| _ -> printfn "No match"
