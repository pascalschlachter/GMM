from lightning.pytorch.cli import LightningCLI

class CustomCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments('data', 'model.init_args.datamodule', apply_on='instantiate')


def main():
    cli = CustomCLI()

if __name__ == '__main__':
    main()
