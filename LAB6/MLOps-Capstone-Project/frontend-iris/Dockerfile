FROM python:3.11-slim 
  
WORKDIR /app 
  
COPY requirements.txt . 
RUN pip install --no-cache-dir -r requirements.txt 
  
COPY . . 
  
EXPOSE 7860 
  
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1
# Variable de entorno con la URL del backend 
# Se sobreescribirá en docker-compose.yml con el nombre de servicio 
ENV BACKEND_URL=http://localhost:8000 
  
CMD ["python", "gradio_app.py"] 
 
 
