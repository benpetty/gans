

install:
	@pipenv install
	@pipenv run sync

dev:
	@pipenv install --dev
	@pipenv run sync

.PHONY:
requirements:
	# warning
	# https://github.com/pypa/pipenv/issues/3493#issuecomment-511741479
	# > It will not work as you expect
	# > It performs an update, which potentially can destroy a distribution
	@pipenv lock -r > requirements.txt
	@pipenv lock -r --dev > requirements-dev.txt
