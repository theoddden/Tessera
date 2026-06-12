use reqwest::Client;
use serde_json::json;

#[tokio::test]
#[ignore = "requires running server on localhost:8080"]
async fn test_generate_endpoint() {
    let client = Client::new();
    
    let request_body = json!({
        "user_id": "test-user-123",
        "context": {
            "description": "Senior litigation associate specializing in IP law",
            "domain": "legal"
        },
        "base_model": "meta-llama/Llama-3-8B",
        "target_rank": 16,
        "response_format": "file"
    });
    
    // This test assumes the server is running on localhost:8080
    // In CI, you'd want to start the server first
    let response = client
        .post("http://localhost:8080/generate")
        .json(&request_body)
        .send()
        .await;
    
    // For now, we'll skip this if server isn't running
    match response {
        Ok(resp) => {
            assert!(resp.status().is_success());
            let body = resp.text().await.unwrap();
            assert!(!body.is_empty());
        }
        Err(_) => {
            // Server not running, skip test
            println!("Skipping test_generate_endpoint: server not running");
        }
    }
}

#[tokio::test]
#[ignore = "requires running server on localhost:8080"]
async fn test_health_endpoint() {
    let client = Client::new();
    
    let response = client
        .get("http://localhost:8080/health")
        .send()
        .await;
    
    match response {
        Ok(resp) => {
            assert!(resp.status().is_success());
            let body: serde_json::Value = resp.json().await.unwrap();
            assert_eq!(body["status"], "healthy");
        }
        Err(_) => {
            println!("Skipping test_health_endpoint: server not running");
        }
    }
}

#[tokio::test]
#[ignore = "requires running server on localhost:8080"]
async fn test_metrics_endpoint() {
    let client = Client::new();
    
    let response = client
        .get("http://localhost:8080/metrics")
        .send()
        .await;
    
    match response {
        Ok(resp) => {
            assert!(resp.status().is_success());
            let body = resp.text().await.unwrap();
            assert!(body.contains("tessera_requests_total"));
        }
        Err(_) => {
            println!("Skipping test_metrics_endpoint: server not running");
        }
    }
}
