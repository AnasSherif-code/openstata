import sys, pathlib, unittest

# Ensure project src is on sys.path
src_path = pathlib.Path(__file__).parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

loader = unittest.TestLoader()
suite = loader.discover('tests')
runner = unittest.TextTestRunner()
result = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
