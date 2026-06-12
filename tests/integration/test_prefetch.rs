use reqwest::Client;
use serde_json::json;

#[tokio::test]
async fn test_prefetch_integration() {
    let client = Client::new();
    
    // Generate multiple adapters to build up history
    for i in 0..5 {
        let request_body = json!({
            "user_id": format!("prefetch-test-user-{}", i),
            "context": {
                "description": format!("Test description {}", i),
                "domain": "test"
            },
            "base_model": "meta-llama/Llama-3-8B",
            "target_rank": 16
        });
        
        let response = client
            .post("http://localhost:8080/generate")
            .json(&request_body)
            .send()
            .await;
        
        match response {
            Ok(resp) => {
                assert!(resp.status().is_success());
            }
            Err(_) => {
                println!("Skipping test_prefetch_integration: server not running");
                return;
            }
        }
    }
    
    // Check health to see if prefetch is running
    let health_response = client
        .get("http://localhost:8080/health")
        .send()
        .await;
    
    match health_response {
        Ok(resp) => {
            assert!(resp.status().is_success());
            let body: serde_json::Value = resp.json().await.unwrap();
            // Prefetch should be running in background
            assert_eq!(body["status"], "healthy");
        }
        Err(_) => {
            println!("Skipping health check in prefetch test: server not running");
        }
    }
}
