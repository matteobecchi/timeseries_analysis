from onion_clustering.onion_uni import OnionUni
from sklearn.utils.estimator_checks import check_estimator


# Define the actual test
def check_things():
    TAU_WINDOW = 10  # time resolution of the analysis

    checks = check_estimator(
        OnionUni(tau_window=TAU_WINDOW), generate_only=True
    )

    for check in checks:
        try:
            check[1](OnionUni(tau_window=TAU_WINDOW))
            print(f"{check[1]}: PASSED")
        except Exception as e:
            print(f"{check[1]}: FAILED\n{e}")


check_things()
