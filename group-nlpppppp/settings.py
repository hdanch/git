import os
import logging.config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGGING = {
	'version': 1,
	'disable_existing_loggers': True,
	'formatters': {
		'verbose': {
                    'class': 'logging.Formatter',
                    'format': '%(asctime)s %(filename)s [line:%(lineno)d] %(levelname)s %(message)s',
		},
		'simple': {
                    'class': 'logging.Formatter',
                    'format': '%(asctime)s %(message)s',
		},
	},
	'handlers': {
		'console':{
			'level':'DEBUG',
			'class':'logging.StreamHandler',
			'formatter': 'verbose'
		},
		'file': {
			'level': 'DEBUG',
			'class': 'logging.FileHandler',
			'filename': BASE_DIR + '/log/record.log',
			'formatter': 'verbose',
		},
	},
	'loggers': {
		'task3': {
			'handlers': ['console', 'file'],
			'level': 'DEBUG',
		},
	}
}

logging.config.dictConfig(LOGGING)
