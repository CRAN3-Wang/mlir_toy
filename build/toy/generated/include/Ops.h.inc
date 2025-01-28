/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Declarations                                                            *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#if defined(GET_OP_CLASSES) || defined(GET_OP_FWD_DEFINES)
#undef GET_OP_FWD_DEFINES
namespace mlir {
namespace toy {
class AddOp;
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {
class ConstantOp;
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {
class FuncOp;
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {
class GenericCallOp;
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {
class MulOp;
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {
class PrintOp;
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {
class ReshapeOp;
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {
class ReturnOp;
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {
class TransposeOp;
} // namespace toy
} // namespace mlir
#endif

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::AddOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class AddOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  AddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  AddOpGenericAdaptorBase(AddOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class AddOpGenericAdaptor : public detail::AddOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::AddOpGenericAdaptorBase;
public:
  AddOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  AddOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : AddOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = AddOp, typename = std::enable_if_t<std::is_same_v<LateInst, AddOp>>>
  AddOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class AddOpAdaptor : public AddOpGenericAdaptor<::mlir::ValueRange> {
public:
  using AddOpGenericAdaptor::AddOpGenericAdaptor;
  AddOpAdaptor(AddOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class AddOp : public ::mlir::Op<AddOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = AddOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = AddOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.add");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getLhs();
  ::mlir::TypedValue<::mlir::TensorType> getRhs();
  ::mlir::MutableOperandRange getLhsMutable();
  ::mlir::MutableOperandRange getRhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lhs, Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::AddOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::ConstantOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ConstantOpGenericAdaptorBase {
public:
  struct Properties {
    using valueTy = ::mlir::DenseElementsAttr;
    valueTy value;

    auto getValue() {
      auto &propStorage = this->value;
      return ::llvm::cast<::mlir::DenseElementsAttr>(propStorage);
    }
    void setValue(const ::mlir::DenseElementsAttr &propValue) {
      this->value = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.value == this->value &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  ConstantOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const Properties &properties = {}, ::mlir::RegionRange regions = {});

  ConstantOpGenericAdaptorBase(ConstantOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes();
  ::mlir::DenseElementsAttr getValueAttr();
  ::mlir::DenseElementsAttr getValue();
};
} // namespace detail
template <typename RangeT>
class ConstantOpGenericAdaptor : public detail::ConstantOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ConstantOpGenericAdaptorBase;
public:
  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const Properties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ConstantOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ConstantOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  template <typename LateInst = ConstantOp, typename = std::enable_if_t<std::is_same_v<LateInst, ConstantOp>>>
  ConstantOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ConstantOpAdaptor : public ConstantOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ConstantOpGenericAdaptor::ConstantOpGenericAdaptor;
  ConstantOpAdaptor(ConstantOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ConstantOp : public ::mlir::Op<ConstantOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ConstantOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ConstantOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("value")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getValueAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getValueAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.constant");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic &()> getDiag);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::DenseElementsAttr getValueAttr();
  ::mlir::DenseElementsAttr getValue();
  void setValueAttr(::mlir::DenseElementsAttr attr);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, double value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::ConstantOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::FuncOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class FuncOpGenericAdaptorBase {
public:
  struct Properties {
    using arg_attrsTy = ::mlir::ArrayAttr;
    arg_attrsTy arg_attrs;

    auto getArgAttrs() {
      auto &propStorage = this->arg_attrs;
      return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(propStorage);
    }
    void setArgAttrs(const ::mlir::ArrayAttr &propValue) {
      this->arg_attrs = propValue;
    }
    using function_typeTy = ::mlir::TypeAttr;
    function_typeTy function_type;

    auto getFunctionType() {
      auto &propStorage = this->function_type;
      return ::llvm::cast<::mlir::TypeAttr>(propStorage);
    }
    void setFunctionType(const ::mlir::TypeAttr &propValue) {
      this->function_type = propValue;
    }
    using res_attrsTy = ::mlir::ArrayAttr;
    res_attrsTy res_attrs;

    auto getResAttrs() {
      auto &propStorage = this->res_attrs;
      return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(propStorage);
    }
    void setResAttrs(const ::mlir::ArrayAttr &propValue) {
      this->res_attrs = propValue;
    }
    using sym_nameTy = ::mlir::StringAttr;
    sym_nameTy sym_name;

    auto getSymName() {
      auto &propStorage = this->sym_name;
      return ::llvm::cast<::mlir::StringAttr>(propStorage);
    }
    void setSymName(const ::mlir::StringAttr &propValue) {
      this->sym_name = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.arg_attrs == this->arg_attrs &&
        rhs.function_type == this->function_type &&
        rhs.res_attrs == this->res_attrs &&
        rhs.sym_name == this->sym_name &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  FuncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const Properties &properties = {}, ::mlir::RegionRange regions = {});

  FuncOpGenericAdaptorBase(FuncOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes();
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getFunctionTypeAttr();
  ::mlir::FunctionType getFunctionType();
  ::mlir::ArrayAttr getArgAttrsAttr();
  ::std::optional< ::mlir::ArrayAttr > getArgAttrs();
  ::mlir::ArrayAttr getResAttrsAttr();
  ::std::optional< ::mlir::ArrayAttr > getResAttrs();
  ::mlir::Region &getBody();
  ::mlir::RegionRange getRegions();
};
} // namespace detail
template <typename RangeT>
class FuncOpGenericAdaptor : public detail::FuncOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::FuncOpGenericAdaptorBase;
public:
  FuncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const Properties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  FuncOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : FuncOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  template <typename LateInst = FuncOp, typename = std::enable_if_t<std::is_same_v<LateInst, FuncOp>>>
  FuncOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class FuncOpAdaptor : public FuncOpGenericAdaptor<::mlir::ValueRange> {
public:
  using FuncOpGenericAdaptor::FuncOpGenericAdaptor;
  FuncOpAdaptor(FuncOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class FuncOp : public ::mlir::Op<FuncOp, ::mlir::OpTrait::OneRegion, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::ZeroOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait, ::mlir::SymbolOpInterface::Trait, ::mlir::CallableOpInterface::Trait, ::mlir::FunctionOpInterface::Trait, ::mlir::OpTrait::IsIsolatedFromAbove> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = FuncOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = FuncOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("arg_attrs"), ::llvm::StringRef("function_type"), ::llvm::StringRef("res_attrs"), ::llvm::StringRef("sym_name")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getArgAttrsAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getArgAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  ::mlir::StringAttr getFunctionTypeAttrName() {
    return getAttributeNameForIndex(1);
  }

  static ::mlir::StringAttr getFunctionTypeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 1);
  }

  ::mlir::StringAttr getResAttrsAttrName() {
    return getAttributeNameForIndex(2);
  }

  static ::mlir::StringAttr getResAttrsAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 2);
  }

  ::mlir::StringAttr getSymNameAttrName() {
    return getAttributeNameForIndex(3);
  }

  static ::mlir::StringAttr getSymNameAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 3);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.func");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  ::mlir::Region &getBody();
  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic &()> getDiag);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::StringAttr getSymNameAttr();
  ::llvm::StringRef getSymName();
  ::mlir::TypeAttr getFunctionTypeAttr();
  ::mlir::FunctionType getFunctionType();
  ::mlir::ArrayAttr getArgAttrsAttr();
  ::std::optional< ::mlir::ArrayAttr > getArgAttrs();
  ::mlir::ArrayAttr getResAttrsAttr();
  ::std::optional< ::mlir::ArrayAttr > getResAttrs();
  void setSymNameAttr(::mlir::StringAttr attr);
  void setSymName(::llvm::StringRef attrValue);
  void setFunctionTypeAttr(::mlir::TypeAttr attr);
  void setFunctionType(::mlir::FunctionType attrValue);
  void setArgAttrsAttr(::mlir::ArrayAttr attr);
  void setResAttrsAttr(::mlir::ArrayAttr attr);
  ::mlir::Attribute removeArgAttrsAttr();
  ::mlir::Attribute removeResAttrsAttr();
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef name, FunctionType type, ArrayRef<NamedAttribute> attrs = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 4 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
  //===------------------------------------------------------------------===//
  // FunctionOpInterface Methods
  //===------------------------------------------------------------------===//

  /// Returns the argument types of this function.
  ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

  /// Returns the result types of this function.
  ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

  /// Returns the region on the current operation that is callable.
  ::mlir::Region *getCallableRegion() { return &getBody(); }
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::FuncOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::GenericCallOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class GenericCallOpGenericAdaptorBase {
public:
  struct Properties {
    using calleeTy = ::mlir::FlatSymbolRefAttr;
    calleeTy callee;

    auto getCallee() {
      auto &propStorage = this->callee;
      return ::llvm::cast<::mlir::FlatSymbolRefAttr>(propStorage);
    }
    void setCallee(const ::mlir::FlatSymbolRefAttr &propValue) {
      this->callee = propValue;
    }
    bool operator==(const Properties &rhs) const {
      return 
        rhs.callee == this->callee &&
        true;
    }
    bool operator!=(const Properties &rhs) const {
      return !(*this == rhs);
    }
  };
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  Properties properties;
  ::mlir::RegionRange odsRegions;
public:
  GenericCallOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const Properties &properties = {}, ::mlir::RegionRange regions = {});

  GenericCallOpGenericAdaptorBase(GenericCallOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  const Properties &getProperties() {
    return properties;
  }

  ::mlir::DictionaryAttr getAttributes();
  ::mlir::FlatSymbolRefAttr getCalleeAttr();
  ::llvm::StringRef getCallee();
};
} // namespace detail
template <typename RangeT>
class GenericCallOpGenericAdaptor : public detail::GenericCallOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::GenericCallOpGenericAdaptorBase;
public:
  GenericCallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const Properties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  GenericCallOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : GenericCallOpGenericAdaptor(values, attrs, (properties ? *properties.as<Properties *>() : Properties{}), regions) {}

  template <typename LateInst = GenericCallOp, typename = std::enable_if_t<std::is_same_v<LateInst, GenericCallOp>>>
  GenericCallOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInputs() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class GenericCallOpAdaptor : public GenericCallOpGenericAdaptor<::mlir::ValueRange> {
public:
  using GenericCallOpGenericAdaptor::GenericCallOpGenericAdaptor;
  GenericCallOpAdaptor(GenericCallOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class GenericCallOp : public ::mlir::Op<GenericCallOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::OpInvariants, ::mlir::BytecodeOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = GenericCallOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = GenericCallOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  using Properties = FoldAdaptor::Properties;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    static ::llvm::StringRef attrNames[] = {::llvm::StringRef("callee")};
    return ::llvm::ArrayRef(attrNames);
  }

  ::mlir::StringAttr getCalleeAttrName() {
    return getAttributeNameForIndex(0);
  }

  static ::mlir::StringAttr getCalleeAttrName(::mlir::OperationName name) {
    return getAttributeNameForIndex(name, 0);
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.generic_call");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInputs();
  ::mlir::MutableOperandRange getInputsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static ::mlir::LogicalResult setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic &()> getDiag);
  static ::mlir::Attribute getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop);
  static llvm::hash_code computePropertiesHash(const Properties &prop);
  static std::optional<mlir::Attribute> getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name);
  static void setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value);
  static void populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs);
  static ::mlir::LogicalResult verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag);
  static ::mlir::LogicalResult readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state);
  void writeProperties(::mlir::DialectBytecodeWriter &writer);
  ::mlir::FlatSymbolRefAttr getCalleeAttr();
  ::llvm::StringRef getCallee();
  void setCalleeAttr(::mlir::FlatSymbolRefAttr attr);
  void setCallee(::llvm::StringRef attrValue);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, StringRef callee, ArrayRef<Value> arguments);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange inputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange inputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef callee, ::mlir::ValueRange inputs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef callee, ::mlir::ValueRange inputs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
private:
  ::mlir::StringAttr getAttributeNameForIndex(unsigned index) {
    return getAttributeNameForIndex((*this)->getName(), index);
  }

  static ::mlir::StringAttr getAttributeNameForIndex(::mlir::OperationName name, unsigned index) {
    assert(index < 1 && "invalid attribute index");
    assert(name.getStringRef() == getOperationName() && "invalid operation name");
    return name.getAttributeNames()[index];
  }

public:
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::GenericCallOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::MulOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class MulOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  MulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  MulOpGenericAdaptorBase(MulOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class MulOpGenericAdaptor : public detail::MulOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::MulOpGenericAdaptorBase;
public:
  MulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  MulOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : MulOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = MulOp, typename = std::enable_if_t<std::is_same_v<LateInst, MulOp>>>
  MulOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getLhs() {
    return (*getODSOperands(0).begin());
  }

  ValueT getRhs() {
    return (*getODSOperands(1).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class MulOpAdaptor : public MulOpGenericAdaptor<::mlir::ValueRange> {
public:
  using MulOpGenericAdaptor::MulOpGenericAdaptor;
  MulOpAdaptor(MulOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class MulOp : public ::mlir::Op<MulOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::NOperands<2>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = MulOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = MulOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.mul");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getLhs();
  ::mlir::TypedValue<::mlir::TensorType> getRhs();
  ::mlir::MutableOperandRange getLhsMutable();
  ::mlir::MutableOperandRange getRhsMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value lhs, Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &p);
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::MulOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::PrintOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class PrintOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  PrintOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  PrintOpGenericAdaptorBase(PrintOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class PrintOpGenericAdaptor : public detail::PrintOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::PrintOpGenericAdaptorBase;
public:
  PrintOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  PrintOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : PrintOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = PrintOp, typename = std::enable_if_t<std::is_same_v<LateInst, PrintOp>>>
  PrintOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class PrintOpAdaptor : public PrintOpGenericAdaptor<::mlir::ValueRange> {
public:
  using PrintOpGenericAdaptor::PrintOpGenericAdaptor;
  PrintOpAdaptor(PrintOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class PrintOp : public ::mlir::Op<PrintOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = PrintOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = PrintOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.print");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
public:
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::PrintOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::ReshapeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReshapeOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ReshapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  ReshapeOpGenericAdaptorBase(ReshapeOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ReshapeOpGenericAdaptor : public detail::ReshapeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReshapeOpGenericAdaptorBase;
public:
  ReshapeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ReshapeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ReshapeOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = ReshapeOp, typename = std::enable_if_t<std::is_same_v<LateInst, ReshapeOp>>>
  ReshapeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReshapeOpAdaptor : public ReshapeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReshapeOpGenericAdaptor::ReshapeOpGenericAdaptor;
  ReshapeOpAdaptor(ReshapeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReshapeOp : public ::mlir::Op<ReshapeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::RankedTensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReshapeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReshapeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.reshape");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::ReshapeOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::ReturnOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class ReturnOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  ReturnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  ReturnOpGenericAdaptorBase(ReturnOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class ReturnOpGenericAdaptor : public detail::ReturnOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::ReturnOpGenericAdaptorBase;
public:
  ReturnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  ReturnOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : ReturnOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = ReturnOp, typename = std::enable_if_t<std::is_same_v<LateInst, ReturnOp>>>
  ReturnOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  RangeT getInput() {
    return getODSOperands(0);
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class ReturnOpAdaptor : public ReturnOpGenericAdaptor<::mlir::ValueRange> {
public:
  using ReturnOpGenericAdaptor::ReturnOpGenericAdaptor;
  ReturnOpAdaptor(ReturnOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class ReturnOp : public ::mlir::Op<ReturnOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::ZeroResults, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::VariadicOperands, ::mlir::OpTrait::HasParent<FuncOp>::Impl, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait, ::mlir::OpTrait::IsTerminator> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = ReturnOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = ReturnOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.return");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::Operation::operand_range getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
  bool hasOperand() { return getNumOperands() != 0; }
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::ReturnOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::TransposeOp declarations
//===----------------------------------------------------------------------===//

namespace detail {
class TransposeOpGenericAdaptorBase {
public:
protected:
  ::mlir::DictionaryAttr odsAttrs;
  ::std::optional<::mlir::OperationName> odsOpName;
  ::mlir::RegionRange odsRegions;
public:
  TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {});

  TransposeOpGenericAdaptorBase(TransposeOp op);

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize);
  ::mlir::DictionaryAttr getAttributes();
};
} // namespace detail
template <typename RangeT>
class TransposeOpGenericAdaptor : public detail::TransposeOpGenericAdaptorBase {
  using ValueT = ::llvm::detail::ValueOfRange<RangeT>;
  using Base = detail::TransposeOpGenericAdaptorBase;
public:
  TransposeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs = nullptr, const ::mlir::EmptyProperties &properties = {}, ::mlir::RegionRange regions = {}) : Base(attrs, properties, regions), odsOperands(values) {}

  TransposeOpGenericAdaptor(RangeT values, ::mlir::DictionaryAttr attrs, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions = {}) : TransposeOpGenericAdaptor(values, attrs, (properties ? *properties.as<::mlir::EmptyProperties *>() : ::mlir::EmptyProperties{}), regions) {}

  template <typename LateInst = TransposeOp, typename = std::enable_if_t<std::is_same_v<LateInst, TransposeOp>>>
  TransposeOpGenericAdaptor(RangeT values, LateInst op) : Base(op), odsOperands(values) {}

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index) {
    return Base::getODSOperandIndexAndLength(index, odsOperands.size());
  }

  RangeT getODSOperands(unsigned index) {
    auto valueRange = getODSOperandIndexAndLength(index);
    return {std::next(odsOperands.begin(), valueRange.first),
             std::next(odsOperands.begin(), valueRange.first + valueRange.second)};
  }

  ValueT getInput() {
    return (*getODSOperands(0).begin());
  }

  RangeT getOperands() {
    return odsOperands;
  }

private:
  RangeT odsOperands;
};
class TransposeOpAdaptor : public TransposeOpGenericAdaptor<::mlir::ValueRange> {
public:
  using TransposeOpGenericAdaptor::TransposeOpGenericAdaptor;
  TransposeOpAdaptor(TransposeOp op);

  ::mlir::LogicalResult verify(::mlir::Location loc);
};
class TransposeOp : public ::mlir::Op<TransposeOp, ::mlir::OpTrait::ZeroRegions, ::mlir::OpTrait::OneResult, ::mlir::OpTrait::OneTypedResult<::mlir::TensorType>::Impl, ::mlir::OpTrait::ZeroSuccessors, ::mlir::OpTrait::OneOperand, ::mlir::OpTrait::OpInvariants, ::mlir::ConditionallySpeculatable::Trait, ::mlir::OpTrait::AlwaysSpeculatableImplTrait, ::mlir::MemoryEffectOpInterface::Trait> {
public:
  using Op::Op;
  using Op::print;
  using Adaptor = TransposeOpAdaptor;
  template <typename RangeT>
  using GenericAdaptor = TransposeOpGenericAdaptor<RangeT>;
  using FoldAdaptor = GenericAdaptor<::llvm::ArrayRef<::mlir::Attribute>>;
  static ::llvm::ArrayRef<::llvm::StringRef> getAttributeNames() {
    return {};
  }

  static constexpr ::llvm::StringLiteral getOperationName() {
    return ::llvm::StringLiteral("toy.transpose");
  }

  std::pair<unsigned, unsigned> getODSOperandIndexAndLength(unsigned index);
  ::mlir::Operation::operand_range getODSOperands(unsigned index);
  ::mlir::TypedValue<::mlir::TensorType> getInput();
  ::mlir::MutableOperandRange getInputMutable();
  std::pair<unsigned, unsigned> getODSResultIndexAndLength(unsigned index);
  ::mlir::Operation::result_range getODSResults(unsigned index);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value input);
  static void build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input);
  static void build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes = {});
  ::mlir::LogicalResult verifyInvariantsImpl();
  ::mlir::LogicalResult verifyInvariants();
  ::mlir::LogicalResult verify();
  static void getCanonicalizationPatterns(::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context);
  static ::mlir::ParseResult parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  void print(::mlir::OpAsmPrinter &_odsPrinter);
  void getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects);
public:
};
} // namespace toy
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::toy::TransposeOp)


#endif  // GET_OP_CLASSES

