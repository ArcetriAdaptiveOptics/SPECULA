import os
import ast
import yaml
import sys

# All the classes are:
# AtmoEvolution.yml          BaseOperation.yml      DataStore.yml      FlaskServer.yml    ImRecCalibrator.yml           
# ModalAnalysisWFS.yml  MultiImRecCalibrator.yml    Slopec.yml
# AtmoInfiniteEvolution.yml  BaseProcessingObj.yml  DisplayServer.yml  FuncGenerator.yml  Integrator.yml                
# ModalAnalysis.yml     ProcessingContainer.yml   ShSlopec.yml           SnCalibrator.yml
# AtmoPropagation.yml        CCD.yml                DM.yml             IdealWFS.yml       LowPassFilter.yml             
# Modalrec.yml          PSF.yml                   ShSubapCalibrator.yml  Vibrations.yml
# AtmoRandomPhase.yml        DataSource.yml         Factory.yml        IirFilter.yml      MirrorCommandsCombinator.yml  
# ModulatedPyramid.yml  PyrSlopec.yml             SH.yml                 WindowedIntegration.yml


# TODO new generator classes not 
exposed_classes = [ 'Source', 'Pupilstop',                    
                    'FuncGenerator', 'BaseOperation', 'AtmoEvolution', 'AtmoInfiniteEvolution', 'AtmoPropagation',
                    'ModulatedPyramid', 'CCD', 'Slopec', 'PyrSlopec', 'Modalrec', 'Integrator', 'IirFilter', 'DM', 'PSF', 'DataStore'                
                  ]

class InitMethodVisitor(ast.NodeVisitor):
    """AST Visitor to extract parameters, inputs, and outputs from an __init__ method."""
    
    def __init__(self):
        self.init_params = {}
        self.param_comments = {}
        self.inputs = {}
        self.outputs = []
    
    def visit_FunctionDef(self, node):
        """Visit the __init__ method and extract parameters, inputs, and outputs."""
        if node.name == "__init__":
            total_params = len(node.args.args) - 1  # Exclude 'self'
            num_defaults = len(node.args.defaults)

            # Extract type hints from function annotations
            annotations = {arg.arg: ast.unparse(arg.annotation) if arg.annotation else None for arg in node.args.args[1:]}  # Skip 'self'

            for i, arg in enumerate(node.args.args[1:]):  # Skip 'self'
                param_name = arg.arg
                param_type = annotations.get(param_name, None)
                is_optional = i >= (total_params - num_defaults)
                
                default_value = "None"
                if is_optional:
                    default_index = i - (total_params - num_defaults)
                    default_node = node.args.defaults[default_index]
                    
                    try:
                        default_value = ast.literal_eval(default_node)
                    except (ValueError, TypeError, AttributeError):
                        if isinstance(default_node, ast.Name):
                            default_value = default_node.id  # Handle names like `np`
                        else:
                            default_value = "None"

                # Construct comment with type and optional status
                comment = "Required" if not is_optional else f"Optional (default={default_value})"
                if param_type:
                    comment += f", type: {param_type}"

                self.init_params[param_name] = default_value
                self.param_comments[param_name] = comment

            # Visit the body of __init__ to extract inputs and outputs
            for statement in node.body:
                self.visit(statement)
    
    def visit_Assign(self, node):
        """Extract input and output specifications from self.inputs and self.outputs assignments."""
        if isinstance(node.targets[0], ast.Subscript):
            target = node.targets[0]
            if isinstance(target.value, ast.Attribute) and target.value.attr in ["inputs", "outputs"]:
                key = target.slice.value if isinstance(target.slice, ast.Constant) else target.slice

                if target.value.attr == "inputs" and isinstance(node.value, ast.Call):
                    # Extract input type from: self.inputs['input_name'] = InputValue(type=InputType)
                    for keyword in node.value.keywords:
                        if keyword.arg == "type":
                            input_type = ast.unparse(keyword.value)
                            self.inputs[key] = input_type
                
                elif target.value.attr == "outputs":
                    # Extract output from: self.outputs['out_value'] = self.out_value
                    self.outputs.append(key)

def extract_class_info(file_path):
    """Extracts class name, __init__ method parameters, default values, and types from a Python file."""
    with open(file_path, "r", encoding="utf-8") as file:
        tree = ast.parse(file.read(), filename=file_path)
    
    class_data = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            if not class_name in exposed_classes:
                continue
            visitor = InitMethodVisitor()
            visitor.visit(node)
            
            class_data.append((class_name, visitor.init_params, visitor.param_comments, visitor.inputs, visitor.outputs))

    return class_data

def generate_yaml(class_name, params, comments, inputs, outputs, output_folder):
    """Generates a YAML file with class information, inputs, and outputs."""
    yaml_path = os.path.join(output_folder, f"{class_name}.yml")
    
    with open(yaml_path, "w", encoding="utf-8") as yaml_file:
        yaml_file.write(f"{class_name}:\n")
        
        # Write constructor parameters
        for param, value in params.items():
            # yaml_file.write(f"  {param}: {value}  # {comments[param]}\n")
            yaml_file.write(f"  {param}: {value}\n")

        # Write inputs
        if inputs:
            yaml_file.write("  inputs:\n")
            for input_name, input_type in inputs.items():
                # yaml_file.write(f"    {input_name}: {input_type}  # InputType\n")
                yaml_file.write(f"    {input_name}: {input_type}\n")

        # Write outputs as a YAML list
        if outputs:
            yaml_file.write(f"  outputs: {outputs}\n")
    
    print(f"Generated YAML: {yaml_path}")

def process_python_files(input_folder, output_folder):
    """Processes all Python files in a directory and generates YAML files."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".py"):
            file_path = os.path.join(input_folder, file_name)
            classes = extract_class_info(file_path)
            
            for class_name, params, comments, inputs, outputs in classes:
                generate_yaml(class_name, params, comments, inputs, outputs, output_folder)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_classes.py <input_folder> <output_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist or is not a directory.")
        sys.exit(1)

    process_python_files(input_folder, output_folder)

