from importlib.metadata import version
import sys
import logging

log = logging.getLogger(__name__)


# Define tool descriptions
TOOL_DESCRIPTIONS = {
    'pca': 'Perform Principal Component Analysis on SNP data',
    'simulation': 'Simulate admixed haplotypes using the OnlineSimulator',
    'admixture_mapping': 'Perform admixture mapping analysis'
}

def show_help():
    """
    Display help information for all available tools.
    """
    print("snputils - Process genomes with ease")
    print("\nAvailable tools:")
    for tool, desc in TOOL_DESCRIPTIONS.items():
        print(f"  {tool:<20} {desc}")
    print("\nFor detailed help on a specific tool, use:")
    print("  snputils <tool> --help")

def main():
    try:
        # Get version information
        snputils_version = version("snputils")
        
        # Parse arguments
        arg_list = tuple(sys.argv)
        
        # Handle help and version flags
        if len(arg_list) == 1 or arg_list[1] in ['-h', '--help']:
            show_help()
            return 0
            
        if arg_list[1] in ['-v', '--version']:
            print(f"snputils version {snputils_version}")
            return 0
            
        # Ensure a tool is specified
        if len(arg_list) < 2:
            log.error("No tool specified")
            show_help()
            return 1
            
        # Dispatch to appropriate tool
        tool = arg_list[1]
        if tool not in TOOL_DESCRIPTIONS:
            log.error(f"Unknown tool: {tool}")
            show_help()
            return 1
            
        # Import and run the selected tool
        if tool == 'pca':
            from . import pca
            return pca.plot_and_save_pca(arg_list[2:])
        elif tool == 'admixture_mapping':
            from . import admixture_mapping
            return admixture_mapping.admixmap(arg_list[2:])
        elif tool == 'simulation':
            from ..simulation import simulator_cli
            return simulator_cli.main(arg_list[2:])
            
    except Exception as e:
        log.error(f"Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
