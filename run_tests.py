import unittest
import sys
import os

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_all_tests():
    print("Running AdaptiveTutor AI Test Suite...")
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\nAll tests passed! Ready for submission.")
        return 0
    else:
        print("\nSome tests failed. Check the logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
