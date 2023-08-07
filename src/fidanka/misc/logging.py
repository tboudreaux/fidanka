import logging


class LoggerManager:
    """
    Singleton class to manage a logger that writes log messages to a file and the console.

    Attributes
    ----------
    _instance : LoggerManager
        The single instance of LoggerManager.

    Methods
    -------
    get_logger(level, flevel, clevel)
        Return the logger object.
    config_logger(filename, flevel, clevel)
        Configure the logger to write to the provided filename and set the log levels.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        """
        Create a new instance of LoggerManager if one doesn't exist, otherwise return the existing instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_logger(*args, **kwargs)
        return cls._instance

    def _init_logger(
        self,
        name="fidanka",
        filename="fidanka.log",
        level=logging.INFO,
        flevel=logging.INFO,
        clevel=logging.WARNING,
    ):
        """
        Initialize the logger.

        Parameters
        ----------
        name : str, optional
            The name of the logger. Defaults to 'fidanka'.
        filename : str, optional
            The name of the file to write log messages to. Defaults to 'fidanka.log'.
        level : int, optional
            The log level for the logger. Defaults to logging.INFO.
        flevel : int, optional
            The log level for the file handler. Defaults to logging.INFO.
        clevel : int, optional
            The log level for the console handler. Defaults to logging.WARNING.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.file_handler = logging.FileHandler(filename)
        self.file_handler.setLevel(flevel)
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(clevel)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
        )
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)

    @classmethod
    def get_logger(cls, name=None):
        """
        Return the logger object.

        Returns
        -------
        logging.Logger
            The logger object.
        """
        return cls._instance.logger

    @classmethod
    def config_logger(
        cls, filename, level=logging.INFO, flevel=logging.INFO, clevel=logging.WARNING
    ):
        """
        Configure the logger to write to the provided filename and set the log levels.

        Parameters
        ----------
        filename : str
            The name of the file to write log messages to.
        level : int, optional
            The log level for the logger. Defaults to logging.INFO.
        flevel : int, optional
            The log level for the file handler. Defaults to logging.INFO.
        clevel : int, optional
            The log level for the console handler. Defaults to logging.WARNING.
        """
        if cls._instance is None:
            cls()
        cls._instance.logger.setLevel(level)
        cls._instance.logger.removeHandler(
            cls._instance.file_handler
        )  # remove the old file handler
        cls._instance.file_handler = logging.FileHandler(
            filename
        )  # create a new file handler
        cls._instance.file_handler.setLevel(flevel)
        cls._instance.file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(funcName)s - %(levelname)s - %(message)s"
            )
        )
        cls._instance.logger.addHandler(
            cls._instance.file_handler
        )  # add the new file handler

        cls._instance.console_handler.setLevel(clevel)
