"""
Streamlit Runner Script
Handles setup and launches the Streamlit app
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("Checking Streamlit dependencies...")
    
    required_packages = [
        'streamlit',
        'plotly',
        'pandas',
        'psutil'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Failed to install dependencies. Please install manually:")
            print("pip install -r requirements.txt")
            return False
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    
    directories = ['checkpoints', 'outputs']
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}/ directory")
    
    print("Directory setup complete!")

def launch_streamlit():
    """Launch the Streamlit app"""
    print("Launching Streamlit app...")
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ])
    except KeyboardInterrupt:
        print("\nStreamlit app stopped.")
    except Exception as e:
        print(f"Error launching Streamlit: {e}")

def main():
    """Main function"""
    print("🚗 AutoLane - Streamlit Deployment")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("Setup failed due to missing dependencies.")
        return
    
    # Setup directories
    setup_directories()
    
    # Launch Streamlit
    print("\n🎉 Setup complete! Launching Streamlit app...")
    print("The app will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the app")
    print("-" * 50)
    
    launch_streamlit()

if __name__ == "__main__":
    main()
