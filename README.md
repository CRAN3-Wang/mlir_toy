# mlir_toy
一个mlir toy tutorial的笔记。
## A Brief Intro. to toy lang
极简介绍toy语言的背景。
### Lexer
Toy语言编译器的基础要从分词器Lexer开始说起。
#### Auxiliary
我们首先声明一些帮助作用的类，结构体，函数。
```cpp
/// Structure definition a location in a file.
struct Location {
  std::shared_ptr<std::string> file; ///< filename.
  int line;                          ///< line number.
  int col;                           ///< column number.
};

// List of Token returned by the lexer.
enum Token : int {
  tok_semicolon = ';',
  tok_parenthese_open = '(',
  tok_parenthese_close = ')',
  tok_bracket_open = '{',
  tok_bracket_close = '}',
  tok_sbracket_open = '[',
  tok_sbracket_close = ']',
  
  tok_eof = -1,
  
  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,
  
  // primary
  tok_identifier = -5,
  tok_number = -6,
};
```
我们首先声明了结构体Location，主要作用是记录读取的.toy文件的文件名（主要是stream）和行列；这样我们就可以对每一个token进行定位，对debug有帮助，
除此之外，我们定义了Token的划分，如果是特殊token我们可以用enum来表示他们，对于剩下的普通文本就按照ASCAII码记录。
#### Lexer
接下来给出Lexer类。
```cpp
// Lexer是抽象基类，提供解析器所需的设施，逐个处理流中的标记并记录文件位置
class Lexer {
public:
  // 为给定文件名创建Lexer，文件名仅用于调试
  Lexer(std::string filename);
  virtual ~Lexer() = default;
  // 查看流中的当前标记
  Token getCurToken();
  // 移动到流中的下一个标记并返回
  Token getNextToken();
  // 移动到流中的下一个标记，断言当前标记符合预期
  void consume(Token tok);
  // 返回当前标识符（前提：getCurToken() == tok_identifier）
  llvm::StringRef getId();
  // 返回当前数字（前提：getCurToken() == tok_number）
  double getValue();
  // 返回当前标记的起始位置
  Location getLastLocation();
  // 返回文件的当前行号
  int getLine();
  // 返回文件的当前列号
  int getCol();
private:
  // 委托派生类获取下一行，返回空字符串表示文件结束
  virtual llvm::StringRef readNextLine() = 0;
  // 从流中返回下一个字符，管理当前行缓冲区
  int getNextChar();
  // 从标准输入返回下一个标记
  Token getTok();
  Token curTok;           // 上次从输入中读取的标记
  Location lastLocation;  // curTok的位置
  std::string identifierStr;  // 当前标记为标识符时，包含其值
  double numVal;          // 当前标记为数字时，包含其值
  Token lastChar;         // getNextChar()上次返回的值
  int curLineNum;         // 输入流的当前行号
  int curCol;             // 输入流的当前列号
  llvm::StringRef curLineBuffer;  // 派生类在readNextLine()调用时提供的缓冲区
};
```
Lexer是一个虚基类，需要在子类实现`virtual llvm::StringRef readNextLine()`才能实例化。
我们关注Lexer类的关键，本质上Lexer只负责每次读取一个token，然后记录。对于不同的输入我们只需要实现不同的readNextline就可以了。解析语法&&控制Lexer的工作由Parser完成。
### Parser
Parser负责理解代码中的语义，并且根据语法规则将代码翻译成AST，下面直接给出一些关键的声明。
```cpp
class Parser {
public:
  // 为提供的Lexer创建Parser
  Parser(Lexer &lexer) : lexer(lexer) {}
  // 解析一个完整的模块，模块是一系列函数定义
  std::unique_ptr<ModuleAST> parseModule();
private:
  Lexer &lexer;
  // 解析返回语句
  std::unique_ptr<ReturnExprAST> parseReturn();
  // 解析字面量数字
  std::unique_ptr<ExprAST> parseNumberExpr();
  // 解析字面量数组表达式
  std::unique_ptr<ExprAST> parseTensorLiteralExpr();
  // 解析括号表达式
  std::unique_ptr<ExprAST> parseParenExpr();
  // 解析标识符表达式
  std::unique_ptr<ExprAST> parseIdentifierExpr();
  // 解析主表达式
  std::unique_ptr<ExprAST> parsePrimary();
  // 递归解析二元运算符的右侧
  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec, std::unique_ptr<ExprAST> lhs);
  // 解析表达式
  std::unique_ptr<ExprAST> parseExpression();
  // 解析类型
  std::unique_ptr<VarType> parseType();
  // 解析变量声明
  std::unique_ptr<VarDeclExprAST> parseDeclaration();
  // 解析代码块
  std::unique_ptr<ExprASTList> parseBlock();
  // 解析原型
  std::unique_ptr<PrototypeAST> parsePrototype();
  // 解析函数定义
  std::unique_ptr<FunctionAST> parseDefinition();
  // 获取待处理二元运算符的优先级
  int getTokPrecedence();
  // 解析错误辅助函数
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "");
};
```
可以看出Parser类有一个变量lexer，这是个Lexer类实例。在private中对不同的token都有对应的parser方法。在public中有parseModule方法，这是这个类的核心。通过调用parseModule，我们控制lexer逐token截取并解析代码。
#Exp
```cpp
def foo(int a, int b){
	int c = a + b
	return c
}
```
对于这个函数，首先创建一个AST，以Module开始。lexer读取第一行得到第一个token是def，解析出我们在定义函数，随后就会将下一个token："foo"记录为id，然后是"("，接下来会解析参数列表。这个过程是由`std::unique_ptr<PrototypeAST> parsePrototype()`完成的，他会调用其他的解析函数帮助解析。接下来就是`std::unique_ptr<ExprASTList> parseBlock()`解析Block，规则基本是相同的。这两个组合起来就是完成了`std::unique_ptr<FunctionAST> parseDefinition()`的一部分工作。实际过程会复杂一些，因为包括语法错误时的断言和处理。
### AST
AST是MLIR中重要的一环，我们通过Parser解析Lexer切割得到的token最后组成不同等级的AST，最后再把这些AST缩进组合得到完整的AST。LLVM为我们提供了AST到IR的转换，所以到AST就已经完成了一个基于LLVM的编译器的框架。
```cpp
struct Indent {
  Indent(int &level) : level(level) { ++level; }
  ~Indent() { --level; }
  int &level;
};

class ASTDumper {
public:
  void dump(ModuleAST *node);

private:
  void dump(const VarType &type);
  void dump(VarDeclExprAST *varDecl);
  void dump(ExprAST *expr);
  void dump(ExprASTList *exprList);
  void dump(NumberExprAST *num);
  void dump(LiteralExprAST *node);
  void dump(VariableExprAST *node);
  void dump(ReturnExprAST *node);
  void dump(BinaryExprAST *node);
  void dump(CallExprAST *node);
  void dump(PrintExprAST *node);
  void dump(PrototypeAST *node);
  void dump(FunctionAST *node);

  void indent();
  int curIndent = 0;
};
```
有一个巧妙的实现，我们通过RAII来控制缩进，每次进入一个低等级的AST，都需要多一个缩进，离开这个等级又要恢复缩进，通过声明周期就可以很好的控制这个过程。
另外我们也有一些dump方法来打印出不同AST的信息。
繁琐的AST实现在这里忽略。
## AST to IR
这是toy tutorial的第二章，也是非常关键的一章。在这一章，我们会了解如何定义自己的dialect和其中的operation，这些工作在tablegen的帮助下会容易一些。首先我们要了解一些基本概念。
MLIR编译系统的核心就是不同的dialect，我们可以将高级的dialect逐级lowering到低级dialect，并且在每一级dialect都可以做对应的优化。另外MLIR自带一些高性能轮子dialect，所以基本上我们只需要把自定义的dialect和Op维护好，剩下的直接接入到自带的dialect中就已经有很好的性能表现了。所以关键就是如何定义dialect和Op并写好pass和transform，在这一章我们着手dialect和Op的定义流程。
### TableGen
#### 注册dialect和Op
我们在这里注册dialect和Op的基类。
```cpp
def Toy_Dialect : Dialect {
  let name = "toy";
  let summary = "A high-level dialect for analyzing and optimizing the Toy language";
  let description = [{
    The Toy language is a tensor-based language that allows you to define
    functions, perform some math computation, and print results. This dialect
    provides a representation of the language that is amenable to analysis and
    optimization.
  }];
  let cppNamespace = "::mlir::toy";
}

class Toy_Op<string mnemonic, list<Trait> traits = []> :
    Op<Toy_Dialect, mnemonic, traits>;
```
#### Op的Operation Definition Specification定义
我们在TableGen通过ODS来声明式的定义Op。
```cpp
def ConstantOp : Toy_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:
    %% ```mlir
      %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ``` %%
  }];
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
  let hasCustomAssemblyFormat = 1;
  let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]},
    OpBuilder<(ins "double":$value)>
  ];
  let hasVerifier = 1;
}
```
其中builders是将一个Op实例化的操作。在这里我们可以对一个Op写多个build方式，如果其中一个OpBuilder显式调用了`build`命令那么我们就不需要再单独写针对这个prototype的build实现。但如果是像`OpBuilder<(ins "double":$value)>`没有直接调用build命令的则需要单独实现。
### 实现
#### 注册
```cpp
/// Dialect initialization, the instance will be owned by the context. This is the point of registration of types and operations for the dialect.
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Ops.cpp.inc"
      >();
}
```
#### ConstOp
在使用tableGen生成代码后，我们需要补充其他实现。
##### build
```cpp
void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  ConstantOp::build(builder, state, dataType, dataAttribute);
}
```
在build里我们需要向builder和state传入所需的信息，在这个Op中就是dataType和dataAttribute，分别对应着Op返回的类型和Op的属性。
具体说就是我们使用`auto dataType = RankedTensorType::get({}, builder.getF64Type())`创建了一个空Type作为Op返回值的标记，然后通过`auto dataAttribute = DenseElementsAttr::get(dataType, value)`得到一个包装value的`DenseElementsAttr`用于表示Op的属性。
#def 属性：描述Op的特性，行为，或静态数据，是数据的metadata。例如我们的数据是作为`DenseElementsAttr`存在的，也可能是是一些其他信息例如flag。
#def 类型：类型是Op输入，输出的类型。
##### parse&print
```cpp
mlir::ParseResult ConstantOp::parse(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes) || parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

void ConstantOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printOptionalAttrDict((*this)->getAttrs(), /*elidedAttrs=*/{"value"});
  printer << getValue();
}
```
我们要自定义parse用于解析自定义Op的MLIR，将“文本型的”MLIR转为程序中的Op。
print做相反的工作，它负责将程序中的Op打印成“文本型的”MLIR。
##### verify
```cpp
mlir::LogicalResult ConstantOp::verify() {
  auto resultType = llvm::dyn_cast<mlir::RankedTensorType>(getResult().getType());
  if (!resultType)
    return success();

  auto attrType = llvm::cast<mlir::RankedTensorType>(getValue().getType());
  if (attrType.getRank() != resultType.getRank()) {
    return emitOpError("return type must match the one of the attached value attribute: ")
           << attrType.getRank() << " != " << resultType.getRank();
  }

  for (int dim = 0, dimE = attrType.getRank(); dim < dimE; ++dim) {
    if (attrType.getShape()[dim] != resultType.getShape()[dim]) {
      return emitOpError("return type shape mismatches its attribute at dimension ")
             << dim << ": " << attrType.getShape()[dim] << " != " << resultType.getShape()[dim];
    }
  }
  return mlir::success();
}
```
verify用于在每次Op发生变化（创建，优化）时依旧是合法的。
### MLIRGen
这一部分主要实现了如何从AST翻译到MLIR的过程。
#### Module
```cpp
theModule = mlir::ModuleOp::create(builder.getUnknownLoc());
```
创建一个空的ModuleOp，作为后面所有Op的顶层容器。
#### FuncOp
由于我们的toy语言是过程式语言，所以文件中只有各种不同的function，所以ModuleAST装有所有的FuncAST，所以我们要遍历每一个FuncAST并生成对应的FuncOp插入到MLIR中。
```cpp
  /// Emit a new function and add it to the MLIR module.
  mlir::toy::FuncOp mlirGen(FunctionAST &funcAST) {
    // Create a scope in the symbol table to hold variable declarations.
    ScopedHashTableScope<llvm::StringRef, mlir::Value> varScope(symbolTable);
  
    // Create an MLIR function for the given prototype.
    builder.setInsertionPointToEnd(theModule.getBody());
    mlir::toy::FuncOp function = mlirGen(*funcAST.getProto());
    if (!function)
      return nullptr;

  }
}
```
这一部分我们创建了**作用域**用于管理这个函数内的变量的生命周期，这个`ScopedHashTableScope`是一个RAII对象，会在生命周期结束时释放内存。
设置Op的插入点移动到Module的尾部，用于插入新的函数声明，随后生成prototype。
```cpp
  

    // Let's start the body of the function now!
    mlir::Block &entryBlock = function.front();
    auto protoArgs = funcAST.getProto()->getArgs();
  
    // Declare all the function arguments in the symbol table.
    for (const auto nameValue :
         llvm::zip(protoArgs, entryBlock.getArguments())) {
      if (failed(declare(std::get<0>(nameValue)->getName(),
                         std::get<1>(nameValue))))
        return nullptr;
    }
```
上面是函数参数列表的生成，首先找到函数体block开始的位置，随后从prototype中获取参数列表。参数列表中的各种变量将被声明并记录在符号表中。
```cpp
    // Set the insertion point in the builder to the beginning of the function
    // body, it will be used throughout the codegen to create operations in this
    // function.
    builder.setInsertionPointToStart(&entryBlock);
  
    // Emit the body of the function.
    if (mlir::failed(mlirGen(*funcAST.getBody()))) {
      function.erase();
      return nullptr;
    }
```
这里我们调用mlirGen生成函数的body部分。
```cpp
    // Implicitly return void if no return statement was emitted.
    // FIXME: we may fix the parser instead to always return the last expression
    // (this would possibly help the REPL case later)
    ReturnOp returnOp;
    if (!entryBlock.empty())
      returnOp = dyn_cast<ReturnOp>(entryBlock.back());
    if (!returnOp) {
      builder.create<ReturnOp>(loc(funcAST.getProto()->loc()));
    } else if (returnOp.hasOperand()) {
      // Otherwise, if this return operation has an operand then add a result to
      // the function.
      function.setType(builder.getFunctionType(
          function.getFunctionType().getInputs(), getType(VarType{})));
    }
    return function;
```
这里如果`entryBlock`不空我们就试图把这个block最后一行动态转换成ReturnOp，随后再根据转换结果判断原block是否含有return语句。如果有返回就更新我们之前创建的FuncOp（function），没有的话创建空返回再更新function。最后生成完所有的FuncOp后我们会调用verify验证module的合法性。
#### ExprOp
```cpp
mlir::Value mlirGen(ExprAST &expr) {
  switch (expr.getKind()) {
  case toy::ExprAST::Expr_BinOp:
    return mlirGen(cast<BinaryExprAST>(expr));
  case toy::ExprAST::Expr_Var:
    return mlirGen(cast<VariableExprAST>(expr));
  case toy::ExprAST::Expr_Literal:
    return mlirGen(cast<LiteralExprAST>(expr));
  case toy::ExprAST::Expr_Call:
    return mlirGen(cast<CallExprAST>(expr));
  case toy::ExprAST::Expr_Num:
    return mlirGen(cast<NumberExprAST>(expr));
  default:
    emitError(loc(expr.loc()))
        << "MLIR codegen encountered an unhandled expr kind '"
        << Twine(expr.getKind()) << "'";
    return nullptr;
  }
}
```
实际上AST中的语句最后都要拆解到**表达式**，然后再根据RTTI推断应该调用哪种ExprAST的mlirGen。
## High-Level Transformation
上一章中我们完成了对自定义dialect和Op的定义，接下来我们的主要工作集中在如何将Op接入到MLIR体系中。在这一部分我们主要关心如何对High-Level语言做转化，换言之写一些简单的pass，而且这些pass都是集中在自定义dialect这个level的。教程中介绍两种方法，一种是cpp实现，另一种是通过tableGen实现声明式的DRR。我们主要介绍DRR因为更加简单且符合设计哲学，缺点是没有那么灵活。
### Declarative Rewrite Rules
若想启用Op的pass，我们需要在主函数中加入`mlir::registerPassManagerCLOptions();`语句注册passes。
#### Basic Pattern-Match and Rewrite
```cpp
// Reshape(Reshape(x)) = Reshape(x)
def ReshapeReshapeOptPattern : Pat<(ReshapeOp(ReshapeOp $arg)),
                                   (ReshapeOp $arg)>;
```
这里我们实现了一个非常简单的重写规则，即将连续的两个reshape简化成一次reshape。这里arg是对要被reshape的对象的占位，猜测ReshapeOp在被build时Type就已经包含了如何reshape的信息，所以这里Op只需要一个arg就可以完成reshape.
#### Pattern-Match and Rewrite using Native Code Call
```cpp
// Reshape(Constant(x)) = x'
def ReshapeConstant :
  NativeCodeCall<"$0.reshape(::llvm::cast<ShapedType>($1.getType()))">;
def FoldConstantReshapeOptPattern : Pat<
  (ReshapeOp:$res (ConstantOp $arg)),
  (ConstantOp (ReshapeConstant $arg, $res))>;
```
在这里我们希望对Reshape一个Constant的过程做优化（先Constant再Reshape需要动态推断对象的类型，这里reshapeOp是动态的，如果我们可以静态确定最后的目标Type，就可以编译时提前完成reshape）。这个过程可以用cpp代码`$0.reshape(::llvm::cast<ShapedType>($1.getType()))`高效实现，所以我们可以自定义一个Op然后插入到DRR模板中。
另外对于需要动态确定类型的reshape(constant(x))操作，编译时无法生成正确的ConstantOp，所以不会被优化。
这被称为常量折叠技术。
#### Pattern-Match and Rewrite with Constraints
```cpp
// Reshape(x) = x, where input and output shapes are identical
def TypesAreIdentical : Constraint<CPred<"$0.getType() == $1.getType()">>;
def RedundantReshapeOptPattern : Pat<
  (ReshapeOp:$res $arg), (replaceWithValue $arg),
  [(TypesAreIdentical $res, $arg)]>;
```
这里我们可以对满足一定限制的Op进行简化。同样的限制也是用cpp代码传入到DRR框架的。
#### DRR definition
```cpp
/// Note: The DRR definition used for defining patterns is shown below:
class Pattern<
    dag sourcePattern, list<dag> resultPatterns,
    list<dag> additionalConstraints = [],
    list<dag> supplementalPatterns = [],
    dag benefitsAdded = (addBenefit 0)>;
```
## Interface in MLIR
在自定义dialect和Op中，我们经常需要使得某些Op都具备通用的方法。在这一章我们关注如何为Op添加形状推断接口，ShapeInference可以被用于遍历MLIR，对那些动态推断类型的Op根据上文推断此处的类型，从而可以在编译时静态确定类型并优化。这是计算图优化的基础。为了更简单的实现这个功能，我们需要先实现inline功能把所有被调用的函数内联到main函数中，这样可以只考虑一个main函数从而简化流程。
### ToyInlinerInterface
#### Inliner
```cpp
struct ToyInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation *call, Operation *callable, bool wouldBeCloned) const final {
    return true;
  }

  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }

  bool isLegalToInline(Region *, Region *, bool, IRMapping &) const final {
    return true;
  }

  void handleTerminator(Operation *op, ArrayRef<Value> valuesToRepl) const final {
    auto returnOp = cast<ReturnOp>(op);
    assert(returnOp.getNumOperands() == valuesToRepl.size());
    for (const auto &it : llvm::enumerate(returnOp.getOperands()))
      valuesToRepl[it.index()].replaceAllUsesWith(it.value());
  }

  Operation *materializeCallConversion(OpBuilder &builder, Value input, Type resultType, Location conversionLoc) const final {
    return builder.create<CastOp>(conversionLoc, resultType, input);
  }
};
```
MLIR为我们提供了inliner功能的接口，我们只需要继承这个类并完成实现即可。主要需要实现的就是`isLegalToInline`, `handleTerminator`, `materializeCallConversion`三个方法。
他的主要功能是在函数调用处将函数调用Op替换成callee的函数体。`handleTerminator`将callee的ReturnOp转换为值流，首先检查返回参数列表是否和需要在caller中被替换的参数是否一样，随后逐个替换。
另外实现的`materializeCallConversion`会在callee接受了不同类型的输入时插入转换Op。
接下来我们需要注册这个接口：
```cpp
void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "src/Ops.cpp.inc"
      >();
  addInterfaces<ToyInlinerInterface>();
}
```
#### Call&Callable
除了inliner本身的实现外，我们还需要让GenericCallOp能识别函数调用的所有信息，以便实现将callee主体插入的功能。
```cpp
/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable GenericCallOp::getCallableForCallee() {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}
  
/// Set the callee for the generic call operation, this is required by the call
/// interface.
void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
  (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}
  
/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range GenericCallOp::getArgOperands() { return getInputs(); }
  
/// Get the argument operands to the called function as a mutable range, this is
/// required by the call interface.
MutableOperandRange GenericCallOp::getArgOperandsMutable() {
  return getInputsMutable();
}
```
这几项都是实现`CallOpInterface`和`CallableOpInterface`接口要求的方法（这里仅展示GenericCallOp的实现中的代码，实际上tableGen文件也需要在声明FuncOp和GenericCallOp的tb中在traits列表中添加对应的接口）。
#### CastOp
```cpp
def CastOp : Toy_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation"; 
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}
```
根据tutorial显示，要想成功内联，我们还需要`CastOp`，因为在其他Op中，接受的输入可能是`tensor<*xf64>`用以适配不同的形状，但是在调用处可能已经推断出了输入的形状从而导致形状不匹配，这时候就需要`CastOp`来转换。
### InferShape
类似的，对于InferShape，我们在td中给需要的Op的traits中加入`DeclareOpInterfaceMethods<ShapeInferenceOpInterface>`，然后实现也非常简单。
```cpp
void MulOp::inferShapes() { getResult().setType(getLhs().getType()); }
```
我们以`MulOp`为例，只需要给Result设置好对应的Type即可。
## Lowering to Affine
在这一步我们需要实现一些pass，将dialect中的Op和Affine（处理循环Op）等内置dialect链接。按道理我们依旧可以通过DRR的形式在tb中制定重写规则，但是tutorial中选择使用纯cpp实现。
### Helper Functions
这里负责将toy dialect中的TensorType转为更底层的MemRefType。并实现了在一个block内实现内存分配和释放的函数。
```cpp
/// 将 RankedTensorType 转换为相应的 MemRefType
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// 插入分配和释放操作
static Value insertAllocAndDealloc(MemRefType type, Location loc, PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}
```
接下来是对循环类的Op进行Lowering到Affine的Op的实现。这里`processIteration`是一个回调函数，它实现了每一次迭代中执行的操作。传入到`lowerOpToLoops`中我们可以通过这个函数构建Affine dialect中的Op并插入到MLIR中。
```cpp
/// 处理迭代循环的函数类型
using LoopIterationFn = function_ref<Value(OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands, PatternRewriter &rewriter, LoopIterationFn processIteration) {
  auto tensorType = cast<RankedTensorType>(*op->result_type_begin());
  auto loc = op->getLoc();

  // 分配并释放结果张量
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // 创建嵌套仿射循环
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc, ivs);
      });

  // 替换当前操作
  rewriter.replaceOp(op, alloc);
}
```
### BinaryOptoLoop
```cpp
template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &b, ValueRange mem, ValueRange ivs) {
                     typename BinaryOp::Adaptor a(mem);
                     auto lhs = b.create<affine::AffineLoadOp>(loc, a.getLhs(), ivs);
                     auto rhs = b.create<affine::AffineLoadOp>(loc, a.getRhs(), ivs);
                     return b.create<LoweredBinaryOp>(loc, lhs, rhs);
                   });
    return success();
  }
};

using AddOpLowering = BinaryOpLowering<toy::AddOp, arith::AddFOp>;
using MulOpLowering = BinaryOpLowering<toy::MulOp, arith::MulFOp>;
```
这里是直接调用`lowerOpToLoops`并传入lambda函数实现转写。
## Lowering to LLVM
和上面一样，我们继续lower所有的Op到LLVM IR并通过JIT解释执行代码。
### Lowering
先声明：
```cpp
namespace {
struct ToyToLLVMLoweringPass
    : public PassWrapper<ToyToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToLLVMLoweringPass)
  
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace
```
再完成核心功能的实现：
```cpp
void ToyToLLVMLoweringPass::runOnOperation() {
  // 定义转换目标，仅针对 LLVM 方言，设置ModuleOp是合法的，也就是说所有的ModuleOp都不会被转换。
  LLVMConversionTarget target(getContext());
  target.addLegalOp<ModuleOp>();

  // 配置类型转换器，将 MemRef 类型映射到 LLVM 表示。
  LLVMTypeConverter typeConverter(&getContext());

  // 填充转换模式，处理 affine、std 等方言的转换。
  RewritePatternSet patterns(&getContext());
  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // 添加对 Toy 方言中 PrintOp 的转换模式，这里我们是手动实现了PrintOpLowering。
  patterns.add<PrintOpLowering>(&getContext());

  // 执行完整转换，确保仅保留合法操作。
  auto module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
```
完成声明，在主函数中直接调用`createLowerToLLVMPass`即可开启这个Pass。
```cpp
/// 创建将 Toy 方言转换为 LLVM 方言的优化 Pass。
std::unique_ptr<mlir::Pass> mlir::toy::createLowerToLLVMPass() {
  return std::make_unique<ToyToLLVMLoweringPass>();
}
```
### JIT
下面表格给出JIT和其他编译方式的对比：

| **类型**     | **编译时机**  | **特点**                | **例子**                |
| ---------- | --------- | --------------------- | --------------------- |
| **AOT 编译** | 运行前（静态编译） | 直接生成机器码，执行快，但缺乏动态优化   | C/C++、Go              |
| **解释执行**   | 运行时逐行解释   | 无需编译，启动快，但执行效率低       | Python、Ruby（早期）       |
| **JIT 编译** | 运行时动态编译   | 结合两者优势：启动快 + 热点代码高效执行 | Java、C#、JavaScript V8 |
要想给toy实现JIT运行功能，我们需要在主函数中加入`runJit`函数：
```cpp
int runJit(mlir::ModuleOp module) {
  // 初始化 LLVM 目标。
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // 注册从 MLIR 到 LLVM IR 的转换，这必须在 JIT 编译之前完成。
  mlir::registerBuiltinDialectTranslation(*module->getContext());
  mlir::registerLLVMDialectTranslation(*module->getContext());

  // 在执行引擎中使用的优化管道。
  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/enableOpt ? 3 : 0, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);

  // 创建一个 MLIR 执行引擎。执行引擎会立即 JIT 编译模块。
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  assert(maybeEngine && "failed to construct an execution engine");
  auto &engine = maybeEngine.get();

  // 调用 JIT 编译的函数。
  auto invocationResult = engine->invokePacked("main");
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
    return -1;
  }

  return 0;
}
```