import logging
import json

def configure_logging():
    rlogger = logging.getLogger()
    rlogger.setLevel(logging.INFO)
    logging.basicConfig(format="%(levelname)-8s\033[1m%(name)-21s\033[0m: %(message)s")
    logging.addLevelName(logging.WARNING, "\033[1;31m{:8}\033[1;0m".format(logging.getLevelName(logging.WARNING)))
    logging.addLevelName(logging.ERROR, "\033[1;35m{:8}\033[1;0m".format(logging.getLevelName(logging.ERROR)))
    logging.addLevelName(logging.INFO, "\033[1;32m{:8}\033[1;0m".format(logging.getLevelName(logging.INFO)))
    logging.addLevelName(logging.DEBUG, "\033[1;34m{:8}\033[1;0m".format(logging.getLevelName(logging.DEBUG)))

def load_config(config_file):
	# TO DO: validate types of entries in the config
	config = json.load(open(config_file, 'r'))
	required_keys = ['classes', 'particles']
	required_particle_keys = ['branches', 'max_length']

	for k in required_keys:
		if k not in config.keys():
			raise KeyError('Pipeline configuration requires key: {}'.format(k))

	for particle_name, particle_info in config['particles'].iteritems():
		if '_' in particle_name:
			raise ValueError('Particle names cannot have _ in them')
		for k in required_particle_keys:
			if k not in particle_info.keys():
				raise KeyError('Particle configuration requires key: {}'.format(k))

	return config