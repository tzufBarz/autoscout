.PHONY: dev

dev:
	@trap 'kill 0' SIGINT; \
	bash backend/start.sh & \
	cd frontend && flutter run & \
	wait