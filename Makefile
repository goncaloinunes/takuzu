all:: src/search.py src/utils.py src/takuzu.py tests/
	tests/test.sh src/takuzu.py tests/public-tests	
	tests/test.sh src/takuzu.py tests/custom-tests
	tests/test.sh src/takuzu.py tests/performance-tests

setup: requirements.txt
	pip install -r requirements.txt

clean:
	rm -rf __pycache__
	rm -f tests/*/*.result

