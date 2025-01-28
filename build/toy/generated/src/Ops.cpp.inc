/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Op Definitions                                                             *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifdef GET_OP_LIST
#undef GET_OP_LIST

::mlir::toy::AddOp,
::mlir::toy::ConstantOp,
::mlir::toy::FuncOp,
::mlir::toy::GenericCallOp,
::mlir::toy::MulOp,
::mlir::toy::PrintOp,
::mlir::toy::ReshapeOp,
::mlir::toy::ReturnOp,
::mlir::toy::TransposeOp
#endif  // GET_OP_LIST

#ifdef GET_OP_CLASSES
#undef GET_OP_CLASSES


//===----------------------------------------------------------------------===//
// Local Utility Method Definitions
//===----------------------------------------------------------------------===//

namespace mlir {
namespace toy {

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_Ops0(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!(((::llvm::isa<::mlir::TensorType>(type))) && ([](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be tensor of 64-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_type_constraint_Ops1(
    ::mlir::Operation *op, ::mlir::Type type, ::llvm::StringRef valueKind,
    unsigned valueIndex) {
  if (!((((::llvm::isa<::mlir::RankedTensorType>(type))) && ((::llvm::cast<::mlir::ShapedType>(type).hasStaticShape()))) && ([](::mlir::Type elementType) { return (elementType.isF64()); }(::llvm::cast<::mlir::ShapedType>(type).getElementType())))) {
    return op->emitOpError(valueKind) << " #" << valueIndex
        << " must be statically shaped tensor of 64-bit float values, but got " << type;
  }
  return ::mlir::success();
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops0(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::DenseFPElementsAttr>(attr) &&::llvm::cast<::mlir::DenseElementsAttr>(attr).getType().getElementType().isF64())))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: 64-bit float elements attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops0(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Ops0(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops1(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::StringAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: string attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops1(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Ops1(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops2(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::TypeAttr>(attr))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(attr).getValue()))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(attr).getValue())))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: type attribute of function type";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops2(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Ops2(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops3(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !(((::llvm::isa<::mlir::ArrayAttr>(attr))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(attr), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: Array of dictionary attributes";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops3(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Ops3(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops4(
    ::mlir::Attribute attr, ::llvm::StringRef attrName, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
  if (attr && !((::llvm::isa<::mlir::FlatSymbolRefAttr>(attr))))
    return getDiag() << "attribute '" << attrName
        << "' failed to satisfy constraint: flat symbol reference attribute";
  return ::mlir::success();
}
static ::mlir::LogicalResult __mlir_ods_local_attr_constraint_Ops4(
    ::mlir::Operation *op, ::mlir::Attribute attr, ::llvm::StringRef attrName) {
  return __mlir_ods_local_attr_constraint_Ops4(attr, attrName, [op]() {
    return op->emitOpError();
  });
}

static ::mlir::LogicalResult __mlir_ods_local_region_constraint_Ops0(
    ::mlir::Operation *op, ::mlir::Region &region, ::llvm::StringRef regionName,
    unsigned regionIndex) {
  if (!((true))) {
    return op->emitOpError("region #") << regionIndex
        << (regionName.empty() ? " " : " ('" + regionName + "') ")
        << "failed to verify constraint: any region";
  }
  return ::mlir::success();
}
} // namespace toy
} // namespace mlir
namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::AddOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
AddOpGenericAdaptorBase::AddOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.add", odsAttrs.getContext());
}

AddOpGenericAdaptorBase::AddOpGenericAdaptorBase(AddOp op) : AddOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> AddOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr AddOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
AddOpAdaptor::AddOpAdaptor(AddOp op) : AddOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult AddOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> AddOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range AddOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> AddOp::getLhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> AddOp::getRhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange AddOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange AddOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> AddOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range AddOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void AddOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void AddOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult AddOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void AddOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::AddOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::ConstantOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ConstantOpGenericAdaptorBase::ConstantOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.constant", odsAttrs.getContext());
}

ConstantOpGenericAdaptorBase::ConstantOpGenericAdaptorBase(ConstantOp op) : ConstantOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> ConstantOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ConstantOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::DenseElementsAttr ConstantOpGenericAdaptorBase::getValueAttr() {
  auto attr = ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
  return attr;
}

::mlir::DenseElementsAttr ConstantOpGenericAdaptorBase::getValue() {
  auto attr = getValueAttr();
  return attr;
}

} // namespace detail
ConstantOpAdaptor::ConstantOpAdaptor(ConstantOp op) : ConstantOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ConstantOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitError(loc, "'toy.constant' op ""requires attribute 'value'");

  if (tblgen_value && !((::llvm::isa<::mlir::DenseFPElementsAttr>(tblgen_value) &&::llvm::cast<::mlir::DenseElementsAttr>(tblgen_value).getType().getElementType().isF64())))
    return emitError(loc, "'toy.constant' op ""attribute 'value' failed to satisfy constraint: 64-bit float elements attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ConstantOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ConstantOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> ConstantOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ConstantOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult ConstantOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic &()> getDiag) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    getDiag() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.value;
       auto attr = dict.get("value");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        getDiag() << "expected key entry for value in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        getDiag() << "Invalid attribute `value` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute ConstantOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.value;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("value",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code ConstantOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.value.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> ConstantOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "value")
      return prop.value;
  return std::nullopt;
}

void ConstantOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "value") {
       prop.value = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.value)>>(value);
       return;
    }
}

void ConstantOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.value) attrs.append("value", prop.value);
}

::mlir::LogicalResult ConstantOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getValueAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Ops0(attr, "value", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult ConstantOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.value)))
    return ::mlir::failure();
  return ::mlir::success();
}

void ConstantOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.value);
}

::mlir::DenseElementsAttr ConstantOp::getValueAttr() {
  return ::llvm::cast<::mlir::DenseElementsAttr>(getProperties().value);
}

::mlir::DenseElementsAttr ConstantOp::getValue() {
  auto attr = getValueAttr();
  return attr;
}

void ConstantOp::setValueAttr(::mlir::DenseElementsAttr attr) {
  (*this)->setAttr(getValueAttrName(), attr);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, DenseElementsAttr value) {
      build(odsBuilder, odsState, value.getType(), value);
    
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::DenseElementsAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  odsState.addTypes(resultType0);
}

void ConstantOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::DenseElementsAttr value) {
  odsState.getOrAddProperties<Properties>().value = value;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ConstantOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 0u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ConstantOp::verifyInvariantsImpl() {
  auto tblgen_value = getProperties().value; (void)tblgen_value;
  if (!tblgen_value) return emitOpError("requires attribute 'value'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops0(*this, tblgen_value, "value")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ConstantOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

void ConstantOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ConstantOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::FuncOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
FuncOpGenericAdaptorBase::FuncOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.func", odsAttrs.getContext());
}

FuncOpGenericAdaptorBase::FuncOpGenericAdaptorBase(FuncOp op) : FuncOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> FuncOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr FuncOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::StringAttr FuncOpGenericAdaptorBase::getSymNameAttr() {
  auto attr = ::llvm::cast<::mlir::StringAttr>(getProperties().sym_name);
  return attr;
}

::llvm::StringRef FuncOpGenericAdaptorBase::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOpGenericAdaptorBase::getFunctionTypeAttr() {
  auto attr = ::llvm::cast<::mlir::TypeAttr>(getProperties().function_type);
  return attr;
}

::mlir::FunctionType FuncOpGenericAdaptorBase::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
}

::mlir::ArrayAttr FuncOpGenericAdaptorBase::getArgAttrsAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(getProperties().arg_attrs);
  return attr;
}

::std::optional< ::mlir::ArrayAttr > FuncOpGenericAdaptorBase::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr FuncOpGenericAdaptorBase::getResAttrsAttr() {
  auto attr = ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(getProperties().res_attrs);
  return attr;
}

::std::optional< ::mlir::ArrayAttr > FuncOpGenericAdaptorBase::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::Region &FuncOpGenericAdaptorBase::getBody() {
  return *odsRegions[0];
}

::mlir::RegionRange FuncOpGenericAdaptorBase::getRegions() {
  return odsRegions;
}

} // namespace detail
FuncOpAdaptor::FuncOpAdaptor(FuncOp op) : FuncOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult FuncOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_arg_attrs = getProperties().arg_attrs; (void)tblgen_arg_attrs;
  auto tblgen_function_type = getProperties().function_type; (void)tblgen_function_type;
  if (!tblgen_function_type) return emitError(loc, "'toy.func' op ""requires attribute 'function_type'");
  auto tblgen_res_attrs = getProperties().res_attrs; (void)tblgen_res_attrs;
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitError(loc, "'toy.func' op ""requires attribute 'sym_name'");

  if (tblgen_sym_name && !((::llvm::isa<::mlir::StringAttr>(tblgen_sym_name))))
    return emitError(loc, "'toy.func' op ""attribute 'sym_name' failed to satisfy constraint: string attribute");

  if (tblgen_function_type && !(((::llvm::isa<::mlir::TypeAttr>(tblgen_function_type))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(tblgen_function_type).getValue()))) && ((::llvm::isa<::mlir::FunctionType>(::llvm::cast<::mlir::TypeAttr>(tblgen_function_type).getValue())))))
    return emitError(loc, "'toy.func' op ""attribute 'function_type' failed to satisfy constraint: type attribute of function type");

  if (tblgen_arg_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_arg_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_arg_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'toy.func' op ""attribute 'arg_attrs' failed to satisfy constraint: Array of dictionary attributes");

  if (tblgen_res_attrs && !(((::llvm::isa<::mlir::ArrayAttr>(tblgen_res_attrs))) && (::llvm::all_of(::llvm::cast<::mlir::ArrayAttr>(tblgen_res_attrs), [&](::mlir::Attribute attr) { return attr && ((::llvm::isa<::mlir::DictionaryAttr>(attr))); }))))
    return emitError(loc, "'toy.func' op ""attribute 'res_attrs' failed to satisfy constraint: Array of dictionary attributes");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> FuncOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range FuncOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

std::pair<unsigned, unsigned> FuncOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range FuncOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::Region &FuncOp::getBody() {
  return (*this)->getRegion(0);
}

::mlir::LogicalResult FuncOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic &()> getDiag) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    getDiag() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.arg_attrs;
       auto attr = dict.get("arg_attrs");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        getDiag() << "expected key entry for arg_attrs in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        getDiag() << "Invalid attribute `arg_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.function_type;
       auto attr = dict.get("function_type");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        getDiag() << "expected key entry for function_type in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        getDiag() << "Invalid attribute `function_type` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.res_attrs;
       auto attr = dict.get("res_attrs");
    if (attr || /*isRequired=*/false) {
      if (!attr) {
        getDiag() << "expected key entry for res_attrs in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        getDiag() << "Invalid attribute `res_attrs` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }

  {
    auto &propStorage = prop.sym_name;
       auto attr = dict.get("sym_name");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        getDiag() << "expected key entry for sym_name in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        getDiag() << "Invalid attribute `sym_name` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute FuncOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.arg_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("arg_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.function_type;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("function_type",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.res_attrs;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("res_attrs",
                                       propStorage));
    }

    {
      const auto &propStorage = prop.sym_name;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("sym_name",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code FuncOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.arg_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.function_type.getAsOpaquePointer()), 
    llvm::hash_value(prop.res_attrs.getAsOpaquePointer()), 
    llvm::hash_value(prop.sym_name.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> FuncOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "arg_attrs")
      return prop.arg_attrs;

    if (name == "function_type")
      return prop.function_type;

    if (name == "res_attrs")
      return prop.res_attrs;

    if (name == "sym_name")
      return prop.sym_name;
  return std::nullopt;
}

void FuncOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "arg_attrs") {
       prop.arg_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.arg_attrs)>>(value);
       return;
    }

    if (name == "function_type") {
       prop.function_type = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.function_type)>>(value);
       return;
    }

    if (name == "res_attrs") {
       prop.res_attrs = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.res_attrs)>>(value);
       return;
    }

    if (name == "sym_name") {
       prop.sym_name = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.sym_name)>>(value);
       return;
    }
}

void FuncOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.arg_attrs) attrs.append("arg_attrs", prop.arg_attrs);

    if (prop.function_type) attrs.append("function_type", prop.function_type);

    if (prop.res_attrs) attrs.append("res_attrs", prop.res_attrs);

    if (prop.sym_name) attrs.append("sym_name", prop.sym_name);
}

::mlir::LogicalResult FuncOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getArgAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Ops3(attr, "arg_attrs", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getFunctionTypeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Ops2(attr, "function_type", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getResAttrsAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Ops3(attr, "res_attrs", getDiag)))
        return ::mlir::failure();
    }

    {
      ::mlir::Attribute attr = attrs.get(getSymNameAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Ops1(attr, "sym_name", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult FuncOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readOptionalAttribute(prop.arg_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.function_type)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readOptionalAttribute(prop.res_attrs)))
    return ::mlir::failure();

  if (::mlir::failed(reader.readAttribute(prop.sym_name)))
    return ::mlir::failure();
  return ::mlir::success();
}

void FuncOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;

  writer.writeOptionalAttribute(prop.arg_attrs);
  writer.writeAttribute(prop.function_type);

  writer.writeOptionalAttribute(prop.res_attrs);
  writer.writeAttribute(prop.sym_name);
}

::mlir::StringAttr FuncOp::getSymNameAttr() {
  return ::llvm::cast<::mlir::StringAttr>(getProperties().sym_name);
}

::llvm::StringRef FuncOp::getSymName() {
  auto attr = getSymNameAttr();
  return attr.getValue();
}

::mlir::TypeAttr FuncOp::getFunctionTypeAttr() {
  return ::llvm::cast<::mlir::TypeAttr>(getProperties().function_type);
}

::mlir::FunctionType FuncOp::getFunctionType() {
  auto attr = getFunctionTypeAttr();
  return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
}

::mlir::ArrayAttr FuncOp::getArgAttrsAttr() {
  return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(getProperties().arg_attrs);
}

::std::optional< ::mlir::ArrayAttr > FuncOp::getArgAttrs() {
  auto attr = getArgAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

::mlir::ArrayAttr FuncOp::getResAttrsAttr() {
  return ::llvm::dyn_cast_or_null<::mlir::ArrayAttr>(getProperties().res_attrs);
}

::std::optional< ::mlir::ArrayAttr > FuncOp::getResAttrs() {
  auto attr = getResAttrsAttr();
  return attr ? ::std::optional< ::mlir::ArrayAttr >(attr) : (::std::nullopt);
}

void FuncOp::setSymNameAttr(::mlir::StringAttr attr) {
  (*this)->setAttr(getSymNameAttrName(), attr);
}

void FuncOp::setSymName(::llvm::StringRef attrValue) {
  (*this)->setAttr(getSymNameAttrName(), ::mlir::Builder((*this)->getContext()).getStringAttr(attrValue));
}

void FuncOp::setFunctionTypeAttr(::mlir::TypeAttr attr) {
  (*this)->setAttr(getFunctionTypeAttrName(), attr);
}

void FuncOp::setFunctionType(::mlir::FunctionType attrValue) {
  (*this)->setAttr(getFunctionTypeAttrName(), ::mlir::TypeAttr::get(attrValue));
}

void FuncOp::setArgAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getArgAttrsAttrName(), attr);
}

void FuncOp::setResAttrsAttr(::mlir::ArrayAttr attr) {
  (*this)->setAttr(getResAttrsAttrName(), attr);
}

::mlir::Attribute FuncOp::removeArgAttrsAttr() {
    auto &attr = getProperties().arg_attrs;
    attr = {};
    return attr;
}

::mlir::Attribute FuncOp::removeResAttrsAttr() {
    auto &attr = getProperties().res_attrs;
    attr = {};
    return attr;
}

::mlir::LogicalResult FuncOp::verifyInvariantsImpl() {
  auto tblgen_arg_attrs = getProperties().arg_attrs; (void)tblgen_arg_attrs;
  auto tblgen_function_type = getProperties().function_type; (void)tblgen_function_type;
  if (!tblgen_function_type) return emitOpError("requires attribute 'function_type'");
  auto tblgen_res_attrs = getProperties().res_attrs; (void)tblgen_res_attrs;
  auto tblgen_sym_name = getProperties().sym_name; (void)tblgen_sym_name;
  if (!tblgen_sym_name) return emitOpError("requires attribute 'sym_name'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops1(*this, tblgen_sym_name, "sym_name")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops2(*this, tblgen_function_type, "function_type")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops3(*this, tblgen_arg_attrs, "arg_attrs")))
    return ::mlir::failure();

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops3(*this, tblgen_res_attrs, "res_attrs")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;

    for (auto &region : ::llvm::MutableArrayRef((*this)->getRegion(0)))
      if (::mlir::failed(__mlir_ods_local_region_constraint_Ops0(*this, region, "body", index++)))
        return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult FuncOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::FuncOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::GenericCallOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
GenericCallOpGenericAdaptorBase::GenericCallOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const Properties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), properties(properties), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.generic_call", odsAttrs.getContext());
}

GenericCallOpGenericAdaptorBase::GenericCallOpGenericAdaptorBase(GenericCallOp op) : GenericCallOpGenericAdaptorBase(op->getDiscardableAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> GenericCallOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr GenericCallOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

::mlir::FlatSymbolRefAttr GenericCallOpGenericAdaptorBase::getCalleeAttr() {
  auto attr = ::llvm::cast<::mlir::FlatSymbolRefAttr>(getProperties().callee);
  return attr;
}

::llvm::StringRef GenericCallOpGenericAdaptorBase::getCallee() {
  auto attr = getCalleeAttr();
  return attr.getValue();
}

} // namespace detail
GenericCallOpAdaptor::GenericCallOpAdaptor(GenericCallOp op) : GenericCallOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult GenericCallOpAdaptor::verify(::mlir::Location loc) {
  auto tblgen_callee = getProperties().callee; (void)tblgen_callee;
  if (!tblgen_callee) return emitError(loc, "'toy.generic_call' op ""requires attribute 'callee'");

  if (tblgen_callee && !((::llvm::isa<::mlir::FlatSymbolRefAttr>(tblgen_callee))))
    return emitError(loc, "'toy.generic_call' op ""attribute 'callee' failed to satisfy constraint: flat symbol reference attribute");
  return ::mlir::success();
}

std::pair<unsigned, unsigned> GenericCallOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range GenericCallOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range GenericCallOp::getInputs() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange GenericCallOp::getInputsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> GenericCallOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range GenericCallOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

::mlir::LogicalResult GenericCallOp::setPropertiesFromAttr(Properties &prop, ::mlir::Attribute attr, ::llvm::function_ref<::mlir::InFlightDiagnostic &()> getDiag) {
  ::mlir::DictionaryAttr dict = ::llvm::dyn_cast<::mlir::DictionaryAttr>(attr);
  if (!dict) {
    getDiag() << "expected DictionaryAttr to set properties";
    return ::mlir::failure();
  }

  {
    auto &propStorage = prop.callee;
       auto attr = dict.get("callee");
    if (attr || /*isRequired=*/true) {
      if (!attr) {
        getDiag() << "expected key entry for callee in DictionaryAttr to set "
                   "Properties.";
        return ::mlir::failure();
      }
      auto convertedAttr = ::llvm::dyn_cast<std::remove_reference_t<decltype(propStorage)>>(attr);
      if (convertedAttr) {
        propStorage = convertedAttr;
      } else {
        getDiag() << "Invalid attribute `callee` in property conversion: " << attr;
        return ::mlir::failure();
      }
    }
  }
  return ::mlir::success();
}

::mlir::Attribute GenericCallOp::getPropertiesAsAttr(::mlir::MLIRContext *ctx, const Properties &prop) {
    ::mlir::SmallVector<::mlir::NamedAttribute> attrs;
    ::mlir::Builder odsBuilder{ctx};

    {
      const auto &propStorage = prop.callee;
      if (propStorage)
        attrs.push_back(odsBuilder.getNamedAttr("callee",
                                       propStorage));
    }

  if (!attrs.empty())
    return odsBuilder.getDictionaryAttr(attrs);
  return {};
}

llvm::hash_code GenericCallOp::computePropertiesHash(const Properties &prop) {
  return llvm::hash_combine(
    llvm::hash_value(prop.callee.getAsOpaquePointer()));
}

std::optional<mlir::Attribute> GenericCallOp::getInherentAttr(::mlir::MLIRContext *ctx, const Properties &prop, llvm::StringRef name) {
    if (name == "callee")
      return prop.callee;
  return std::nullopt;
}

void GenericCallOp::setInherentAttr(Properties &prop, llvm::StringRef name, mlir::Attribute value) {
    if (name == "callee") {
       prop.callee = ::llvm::dyn_cast_or_null<std::remove_reference_t<decltype(prop.callee)>>(value);
       return;
    }
}

void GenericCallOp::populateInherentAttrs(::mlir::MLIRContext *ctx, const Properties &prop, ::mlir::NamedAttrList &attrs) {
    if (prop.callee) attrs.append("callee", prop.callee);
}

::mlir::LogicalResult GenericCallOp::verifyInherentAttrs(::mlir::OperationName opName, ::mlir::NamedAttrList &attrs, llvm::function_ref<::mlir::InFlightDiagnostic()> getDiag) {
    {
      ::mlir::Attribute attr = attrs.get(getCalleeAttrName(opName));
      if (attr && ::mlir::failed(__mlir_ods_local_attr_constraint_Ops4(attr, "callee", getDiag)))
        return ::mlir::failure();
    }
    return ::mlir::success();
}

::mlir::LogicalResult GenericCallOp::readProperties(::mlir::DialectBytecodeReader &reader, ::mlir::OperationState &state) {
  auto &prop = state.getOrAddProperties<Properties>(); (void)prop;
  if (::mlir::failed(reader.readAttribute(prop.callee)))
    return ::mlir::failure();
  return ::mlir::success();
}

void GenericCallOp::writeProperties(::mlir::DialectBytecodeWriter &writer) {
  auto &prop = getProperties(); (void)prop;
  writer.writeAttribute(prop.callee);
}

::mlir::FlatSymbolRefAttr GenericCallOp::getCalleeAttr() {
  return ::llvm::cast<::mlir::FlatSymbolRefAttr>(getProperties().callee);
}

::llvm::StringRef GenericCallOp::getCallee() {
  auto attr = getCalleeAttr();
  return attr.getValue();
}

void GenericCallOp::setCalleeAttr(::mlir::FlatSymbolRefAttr attr) {
  (*this)->setAttr(getCalleeAttrName(), attr);
}

void GenericCallOp::setCallee(::llvm::StringRef attrValue) {
  (*this)->setAttr(getCalleeAttrName(), ::mlir::SymbolRefAttr::get(::mlir::Builder((*this)->getContext()).getContext(), attrValue));
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().callee = callee;
  odsState.addTypes(resultType0);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::FlatSymbolRefAttr callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().callee = callee;
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::llvm::StringRef callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().callee = ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), callee);
  odsState.addTypes(resultType0);
}

void GenericCallOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::llvm::StringRef callee, ::mlir::ValueRange inputs) {
  odsState.addOperands(inputs);
  odsState.getOrAddProperties<Properties>().callee = ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), callee);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void GenericCallOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult GenericCallOp::verifyInvariantsImpl() {
  auto tblgen_callee = getProperties().callee; (void)tblgen_callee;
  if (!tblgen_callee) return emitOpError("requires attribute 'callee'");

  if (::mlir::failed(__mlir_ods_local_attr_constraint_Ops4(*this, tblgen_callee, "callee")))
    return ::mlir::failure();
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult GenericCallOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult GenericCallOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::FlatSymbolRefAttr calleeAttr;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputsOperands;
  ::llvm::SMLoc inputsOperandsLoc;
  (void)inputsOperandsLoc;
  ::llvm::ArrayRef<::mlir::Type> inputsTypes;
  ::llvm::ArrayRef<::mlir::Type> allResultTypes;

  if (parser.parseCustomAttributeWithFallback(calleeAttr, parser.getBuilder().getType<::mlir::NoneType>())) {
    return ::mlir::failure();
  }
  if (calleeAttr) result.getOrAddProperties<GenericCallOp::Properties>().callee = calleeAttr;
  if (parser.parseLParen())
    return ::mlir::failure();

  inputsOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputsOperands))
    return ::mlir::failure();
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
    if (failed(verifyInherentAttrs(result.name, result.attributes, [&]() {
        return parser.emitError(loc) << "'" << result.name.getStringRef() << "' op ";
      })))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  ::mlir::FunctionType inputs__allResult_functionType;
  if (parser.parseType(inputs__allResult_functionType))
    return ::mlir::failure();
  inputsTypes = inputs__allResult_functionType.getInputs();
  allResultTypes = inputs__allResult_functionType.getResults();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputsOperands, inputsTypes, inputsOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void GenericCallOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter.printAttributeWithoutType(getCalleeAttr());
  _odsPrinter << "(";
  _odsPrinter << getInputs();
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("callee");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  _odsPrinter.printFunctionalType(getInputs().getTypes(), getOperation()->getResultTypes());
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::GenericCallOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::MulOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
MulOpGenericAdaptorBase::MulOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.mul", odsAttrs.getContext());
}

MulOpGenericAdaptorBase::MulOpGenericAdaptorBase(MulOp op) : MulOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> MulOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr MulOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
MulOpAdaptor::MulOpAdaptor(MulOp op) : MulOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult MulOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> MulOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range MulOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> MulOp::getLhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::TypedValue<::mlir::TensorType> MulOp::getRhs() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(1).begin());
}

::mlir::MutableOperandRange MulOp::getLhsMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

::mlir::MutableOperandRange MulOp::getRhsMutable() {
  auto range = getODSOperandIndexAndLength(1);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> MulOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range MulOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  odsState.addTypes(resultType0);
}

void MulOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value lhs, ::mlir::Value rhs) {
  odsState.addOperands(lhs);
  odsState.addOperands(rhs);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void MulOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 2u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult MulOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
    auto valueGroup1 = getODSOperands(1);

    for (auto v : valueGroup1) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult MulOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

void MulOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::MulOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::PrintOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
PrintOpGenericAdaptorBase::PrintOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.print", odsAttrs.getContext());
}

PrintOpGenericAdaptorBase::PrintOpGenericAdaptorBase(PrintOp op) : PrintOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> PrintOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr PrintOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
PrintOpAdaptor::PrintOpAdaptor(PrintOp op) : PrintOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult PrintOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> PrintOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range PrintOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> PrintOp::getInput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange PrintOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> PrintOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range PrintOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Value input) {
  odsState.addOperands(input);
}

void PrintOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 0u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void PrintOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult PrintOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult PrintOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult PrintOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawTypes[0] = type;
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void PrintOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << ' ';
  _odsPrinter << getInput();
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::PrintOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::ReshapeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReshapeOpGenericAdaptorBase::ReshapeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.reshape", odsAttrs.getContext());
}

ReshapeOpGenericAdaptorBase::ReshapeOpGenericAdaptorBase(ReshapeOp op) : ReshapeOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> ReshapeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr ReshapeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReshapeOpAdaptor::ReshapeOpAdaptor(ReshapeOp op) : ReshapeOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ReshapeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReshapeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range ReshapeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> ReshapeOp::getInput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange ReshapeOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReshapeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReshapeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(resultType0);
}

void ReshapeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void ReshapeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReshapeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops1(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReshapeOp::verifyInvariants() {
  return verifyInvariantsImpl();
}

::mlir::ParseResult ReshapeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReshapeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getInput();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void ReshapeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ReshapeOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::ReturnOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
ReturnOpGenericAdaptorBase::ReturnOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.return", odsAttrs.getContext());
}

ReturnOpGenericAdaptorBase::ReturnOpGenericAdaptorBase(ReturnOp op) : ReturnOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> ReturnOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (odsOperandsSize - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::DictionaryAttr ReturnOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
ReturnOpAdaptor::ReturnOpAdaptor(ReturnOp op) : ReturnOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult ReturnOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> ReturnOp::getODSOperandIndexAndLength(unsigned index) {
  bool isVariadic[] = {true};
  int prevVariadicCount = 0;
  for (unsigned i = 0; i < index; ++i)
    if (isVariadic[i]) ++prevVariadicCount;

  // Calculate how many dynamic values a static variadic operand corresponds to.
  // This assumes all static variadic operands have the same dynamic value count.
  int variadicSize = (getOperation()->getNumOperands() - 0) / 1;
  // `index` passed in as the parameter is the static index which counts each
  // operand (variadic or not) as size 1. So here for each previous static variadic
  // operand, we need to offset by (variadicSize - 1) to get where the dynamic
  // value pack for this static operand starts.
  int start = index + (variadicSize - 1) * prevVariadicCount;
  int size = isVariadic[index] ? variadicSize : 1;
  return {start, size};
}

::mlir::Operation::operand_range ReturnOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::Operation::operand_range ReturnOp::getInput() {
  return getODSOperands(0);
}

::mlir::MutableOperandRange ReturnOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> ReturnOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range ReturnOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState) {
 build(odsBuilder, odsState, std::nullopt); 
}

void ReturnOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::ValueRange input) {
  odsState.addOperands(input);
}

void ReturnOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 0u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult ReturnOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult ReturnOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult ReturnOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4> inputOperands;
  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::llvm::SmallVector<::mlir::Type, 1> inputTypes;

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperandList(inputOperands))
    return ::mlir::failure();
  if (!inputOperands.empty()) {
  if (parser.parseColon())
    return ::mlir::failure();

  if (parser.parseTypeList(inputTypes))
    return ::mlir::failure();
  }
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void ReturnOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  if (!getInput().empty()) {
    _odsPrinter << ' ';
    _odsPrinter << getInput();
    _odsPrinter << ' ' << ":";
    _odsPrinter << ' ';
    _odsPrinter << getInput().getTypes();
  }
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

void ReturnOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::ReturnOp)

namespace mlir {
namespace toy {

//===----------------------------------------------------------------------===//
// ::mlir::toy::TransposeOp definitions
//===----------------------------------------------------------------------===//

namespace detail {
TransposeOpGenericAdaptorBase::TransposeOpGenericAdaptorBase(::mlir::DictionaryAttr attrs, const ::mlir::EmptyProperties &properties, ::mlir::RegionRange regions) : odsAttrs(attrs), odsRegions(regions) {  if (odsAttrs)
    odsOpName.emplace("toy.transpose", odsAttrs.getContext());
}

TransposeOpGenericAdaptorBase::TransposeOpGenericAdaptorBase(TransposeOp op) : TransposeOpGenericAdaptorBase(op->getAttrDictionary(), op.getProperties(), op->getRegions()) {}

std::pair<unsigned, unsigned> TransposeOpGenericAdaptorBase::getODSOperandIndexAndLength(unsigned index, unsigned odsOperandsSize) {
  return {index, 1};
}

::mlir::DictionaryAttr TransposeOpGenericAdaptorBase::getAttributes() {
  return odsAttrs;
}

} // namespace detail
TransposeOpAdaptor::TransposeOpAdaptor(TransposeOp op) : TransposeOpGenericAdaptor(op->getOperands(), op) {}

::mlir::LogicalResult TransposeOpAdaptor::verify(::mlir::Location loc) {
  return ::mlir::success();
}

std::pair<unsigned, unsigned> TransposeOp::getODSOperandIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::operand_range TransposeOp::getODSOperands(unsigned index) {
  auto valueRange = getODSOperandIndexAndLength(index);
  return {std::next(getOperation()->operand_begin(), valueRange.first),
           std::next(getOperation()->operand_begin(), valueRange.first + valueRange.second)};
}

::mlir::TypedValue<::mlir::TensorType> TransposeOp::getInput() {
  return ::llvm::cast<::mlir::TypedValue<::mlir::TensorType>>(*getODSOperands(0).begin());
}

::mlir::MutableOperandRange TransposeOp::getInputMutable() {
  auto range = getODSOperandIndexAndLength(0);
  auto mutableRange = ::mlir::MutableOperandRange(getOperation(), range.first, range.second);
  return mutableRange;
}

std::pair<unsigned, unsigned> TransposeOp::getODSResultIndexAndLength(unsigned index) {
  return {index, 1};
}

::mlir::Operation::result_range TransposeOp::getODSResults(unsigned index) {
  auto valueRange = getODSResultIndexAndLength(index);
  return {std::next(getOperation()->result_begin(), valueRange.first),
           std::next(getOperation()->result_begin(), valueRange.first + valueRange.second)};
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::Type resultType0, ::mlir::Value input) {
  odsState.addOperands(input);
  odsState.addTypes(resultType0);
}

void TransposeOp::build(::mlir::OpBuilder &odsBuilder, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::Value input) {
  odsState.addOperands(input);
  assert(resultTypes.size() == 1u && "mismatched number of results");
  odsState.addTypes(resultTypes);
}

void TransposeOp::build(::mlir::OpBuilder &, ::mlir::OperationState &odsState, ::mlir::TypeRange resultTypes, ::mlir::ValueRange operands, ::llvm::ArrayRef<::mlir::NamedAttribute> attributes) {
  assert(operands.size() == 1u && "mismatched number of parameters");
  odsState.addOperands(operands);
  odsState.addAttributes(attributes);
  assert(resultTypes.size() == 1u && "mismatched number of return types");
  odsState.addTypes(resultTypes);
}

::mlir::LogicalResult TransposeOp::verifyInvariantsImpl() {
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSOperands(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "operand", index++)))
        return ::mlir::failure();
    }
  }
  {
    unsigned index = 0; (void)index;
    auto valueGroup0 = getODSResults(0);

    for (auto v : valueGroup0) {
      if (::mlir::failed(__mlir_ods_local_type_constraint_Ops0(*this, v.getType(), "result", index++)))
        return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

::mlir::LogicalResult TransposeOp::verifyInvariants() {
  if(::mlir::succeeded(verifyInvariantsImpl()) && ::mlir::succeeded(verify()))
    return ::mlir::success();
  return ::mlir::failure();
}

::mlir::ParseResult TransposeOp::parse(::mlir::OpAsmParser &parser, ::mlir::OperationState &result) {
  ::mlir::OpAsmParser::UnresolvedOperand inputRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> inputOperands(inputRawOperands);  ::llvm::SMLoc inputOperandsLoc;
  (void)inputOperandsLoc;
  ::mlir::Type inputRawTypes[1];
  ::llvm::ArrayRef<::mlir::Type> inputTypes(inputRawTypes);
  ::llvm::SmallVector<::mlir::Type, 1> allResultTypes;
  if (parser.parseLParen())
    return ::mlir::failure();

  inputOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(inputRawOperands[0]))
    return ::mlir::failure();
  if (parser.parseColon())
    return ::mlir::failure();

  {
    ::mlir::TensorType type;
    if (parser.parseCustomTypeWithFallback(type))
      return ::mlir::failure();
    inputRawTypes[0] = type;
  }
  if (parser.parseRParen())
    return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();(void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  if (parser.parseKeyword("to"))
    return ::mlir::failure();

  if (parser.parseTypeList(allResultTypes))
    return ::mlir::failure();
  result.addTypes(allResultTypes);
  if (parser.resolveOperands(inputOperands, inputTypes, inputOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

void TransposeOp::print(::mlir::OpAsmPrinter &_odsPrinter) {
  _odsPrinter << "(";
  _odsPrinter << getInput();
  _odsPrinter << ' ' << ":";
  _odsPrinter << ' ';
  {
    auto type = getInput().getType();
    if (auto validType = ::llvm::dyn_cast<::mlir::TensorType>(type))
      _odsPrinter.printStrippedAttrOrType(validType);
   else
     _odsPrinter << type;
  }
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
  _odsPrinter << ' ' << "to";
  _odsPrinter << ' ';
  _odsPrinter << getOperation()->getResultTypes();
}

void TransposeOp::getEffects(::llvm::SmallVectorImpl<::mlir::SideEffects::EffectInstance<::mlir::MemoryEffects::Effect>> &effects) {
}

} // namespace toy
} // namespace mlir
MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::toy::TransposeOp)


#endif  // GET_OP_CLASSES

