#!/usr/bin/env python3
"""
Startup script for Stock Price Predictor
This script helps you start both the backend and frontend
"""

import subprocess
import sys
import time
import webbrowser
import os
import platform

def check_python_dependencies():
    """Check if required Python packages are installed"""
    try:
        import flask
        import pandas
        import numpy
        import sklearn
        import yfinance
        import nltk
        import vaderSentiment
        print("✅ All Python dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False

def check_node_dependencies():
    """Check if Node.js dependencies are installed"""
    if not os.path.exists("node_modules"):
        print("❌ Node.js dependencies not found")
        print("Please run: npm install")
        return False
    print("✅ Node.js dependencies are installed")
    return True

def find_npm_command():
    """Find the correct npm command for the current system"""
    # Common npm command variations
    npm_commands = ["npm", "npm.cmd", "npm.exe"]
    
    for cmd in npm_commands:
        try:
            # Check if command exists
            result = subprocess.run([cmd, "--version"], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if result.returncode == 0:
                print(f"✅ Found npm: {cmd}")
                return cmd
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue
    
    return None

def start_backend():
    """Start the Flask backend server"""
    print("🚀 Starting Flask backend server...")
    try:
        # Start backend in a subprocess
        backend_process = subprocess.Popen([
            sys.executable, "stock_predictor.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        # Check if server is running
        if backend_process.poll() is None:
            print("✅ Backend server started successfully on http://localhost:5000")
            return backend_process
        else:
            stdout, stderr = backend_process.communicate()
            print(f"❌ Backend failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the React frontend"""
    print("🌐 Starting React frontend...")
    
    # Find npm command
    npm_cmd = find_npm_command()
    if not npm_cmd:
        print("❌ npm command not found!")
        print("\n🔧 Solutions:")
        print("1. Install Node.js from: https://nodejs.org/")
        print("2. Make sure Node.js is added to your PATH")
        print("3. Restart your terminal after installation")
        print("4. Or run manually: npm start")
        return None
    
    try:
        # Start frontend in a subprocess
        frontend_process = subprocess.Popen([
            npm_cmd, "start"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for server to start
        time.sleep(5)
        
        # Check if server is running
        if frontend_process.poll() is None:
            print("✅ Frontend started successfully on http://localhost:3000")
            return frontend_process
        else:
            stdout, stderr = frontend_process.communicate()
            print(f"❌ Frontend failed to start: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("🎯 Stock Price Predictor - Startup Script")
    print("=" * 50)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    
    # Check dependencies
    if not check_python_dependencies():
        return
    
    if not check_node_dependencies():
        return
    
    print("\n📋 Starting application...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n💡 Manual Frontend Start:")
        print("Since the frontend couldn't be started automatically:")
        print("1. Open a new terminal/command prompt")
        print("2. Navigate to this directory")
        print("3. Run: npm start")
        print("4. The frontend will open in your browser")
        
        # Keep backend running
        print("\n📊 Backend is still running on http://localhost:5000")
        print("⏳ Press Ctrl+C to stop the backend...")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Shutting down backend...")
            if backend_process:
                backend_process.terminate()
            print("✅ Backend stopped successfully")
        return
    
    print("\n🎉 Application started successfully!")
    print("📊 Backend: http://localhost:5000")
    print("🌐 Frontend: http://localhost:3000")
    print("\n💡 Tips:")
    print("- The frontend will automatically open in your browser")
    print("- Try entering a stock ticker like 'AAPL', 'MSFT', or 'NVDA'")
    print("- Press Ctrl+C to stop both servers")
    
    try:
        # Open browser
        webbrowser.open('http://localhost:3000')
        
        # Keep the script running
        print("\n⏳ Press Ctrl+C to stop the application...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down application...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("✅ Application stopped successfully")

if __name__ == "__main__":
    main()
