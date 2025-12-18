"""
Simple entry point for running electricity market training.

This demonstrates the new structure with the generic MARL framework
in easy_marl.core and the electricity market example in
easy_marl.examples.electricity_market.
"""

from easy_marl.examples.electricity_market.training import auto_train


def main():
    print("Hello from easy_marl!")
    print("Running electricity market MARL training example...")

    auto_train()


if __name__ == "__main__":
    main()
