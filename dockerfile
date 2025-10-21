FROM macroverse-cn-beijing.cr.volces.com/macroverse/ai-agent-base:v1

WORKDIR /root/deepddg

COPY . .

RUN uv sync

ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]
