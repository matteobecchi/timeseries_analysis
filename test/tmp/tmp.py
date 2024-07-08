from onion_clustering.onion_uni import OnionUni
from sklearn.utils.estimator_checks import check_estimator


# Define the actual test
def check_things():
    checks = check_estimator(OnionUni(), generate_only=True)

    for check in checks:
        try:
            function_name = getattr(check[1], "__name__", str(check[1]))
            check[1](OnionUni())
            print(f"{function_name}: PASSED")
        except Exception as e:
            print(f"{function_name}: FAILED\n{e}")


check_things()
