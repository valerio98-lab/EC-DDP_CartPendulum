from multiprocessing import Process
import argparse as arg
from ec_ddp.src import ec_ddp
import ddp
import logging


def run_simulation(main, model_arg):
    main(model_arg)


if __name__ == "__main__":
    required = False
    parser = arg.ArgumentParser(
        description="Run both implementations. Select a model to run using the --model flag or --ddp_model flag and--ec_ddp_model flag if you want to run different models"
    )
    parser.add_argument("--model", type=str, choices=["cart_pendulum", "pendubot"])
    parser.add_argument("--ddp_model", type=str, choices=["cart_pendulum", "pendubot"])
    parser.add_argument("--ec_ddp_model", type=str, choices=["cart_pendulum", "pendubot"])

    args = parser.parse_args()
    if args.model is None and args.ddp_model is None and args.ec_ddp_model is None:
        logging.basicConfig(level=logging.INFO)
        logging.info(
            "Select a model to run using the --model flag or --ddp_model and --ec_ddp_model flags if you want to run different models"
        )
        exit()

    arg_ddp = args.ddp_model if args.model is None else args.model
    arg_ec_ddp = args.ec_ddp_model if args.model is None else args.model

    p1 = Process(target=run_simulation, args=(ec_ddp.main, arg_ec_ddp))
    p2 = Process(target=run_simulation, args=(ddp.main, arg_ddp))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
