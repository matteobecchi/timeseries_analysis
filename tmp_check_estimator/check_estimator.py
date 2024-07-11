"""
Temporary script to run the sklearn check_estimator().

https://scikit-learn.org/stable/modules/generated/
    sklearn.utils.estimator_checks.check_estimator.html
"""

from onion_clustering.onion_uni import OnionUni
from sklearn.utils.estimator_checks import check_estimator


def run_the_checks():
    """Run check_estimator and return the results."""
    _ = check_estimator(OnionUni())
    # checks = check_estimator(OnionUni(), generate_only=True)

    # for check in checks:
    #     try:
    #         function_name = getattr(check[1], "__name__", str(check[1]))
    #         check[1](OnionUni())
    #         print(f"{function_name}: PASSED")
    #     except Exception as e:
    #         print(f"{function_name}: FAILED\n{e}")


run_the_checks()
