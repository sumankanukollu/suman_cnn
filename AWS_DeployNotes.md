# AWS Deployment notes:

### [Serverless Documentation](https://www.serverless.com/framework/docs/providers/aws/)

### [AWS Lambda guide](https://www.serverless.com/aws-lambda)

# Serverless Computing:

- Serverless allows you to build and run applications and services without thinking about servers. 
- Cloud-based computing model.
- It eliminates (hides) infrastructure management tasks such as:
  - server or cluster provisioning
  - operating system maintenance
  - capacity provisioning
  - downtime 
- The unit of execution is often a function:
  - FaaS - Function as a Service
- Pricing is based on the actual amount of resources consumed
- You can build them for nearly any type of application or backend service, and **everything required to run and scale your application with high availability is handled for you!**



# AWS Lambda:

- AWS Lambda lets you run code without provisioning or managing servers. 
- You pay only for the compute time you consume - there is no charge when your code is not running
- It automatically scales your application by running code in response to each trigger. Your code runs in parallel and process each trigger individually, scaling precisely with the size of the workload
- You are charged for every 100ms your code executes
- Can be triggered by different events (e.g. HTTP request)
- Your function code and it's dependencies shouldn't be greater than 250MB (deployment package .zip file)

![](\src\Lambda-RealTimeFileProcessing.jpg)



- Runs on Amazon Linux OS, inside a runtime container
- The more RAM memory you allocate, the faster CPU you get:
  - from 128MB to 3008 MB, in 64 increments
- Max. 500MB of storage inside /tmp directory (where your code/files can expand)
- Free Tier:
  - 1 million requests per month ($0.0000002 per requests thereafter)
  - 400,000 GB-seconds of compute time per month ($0.000000208 for every 100ms used thereafter, 128 MB of RAM)

 # WHAT IS SERVERLESS FRAMEWORK?

The **Serverless Framework** is a free and [open-source (Links to an external site.)](https://en.wikipedia.org/wiki/Open_Source) [web framework (Links to an external site.)](https://en.wikipedia.org/wiki/Web_framework) written using [Node.js (Links to an external site.)](https://en.wikipedia.org/wiki/Node.js). Serverless is the first framework developed for building applications on [AWS Lambda (Links to an external site.)](https://en.wikipedia.org/wiki/Amazon_Lambda), a [serverless computing (Links to an external site.)](https://en.wikipedia.org/wiki/Serverless_computing) platform provided by [Amazon (Links to an external site.)](https://en.wikipedia.org/wiki/Amazon.com) as a part of [Amazon Web Services (Links to an external site.)](https://en.wikipedia.org/wiki/Amazon_Web_Services) but now can work with others as well.

 

- A Serverless app can simply be a couple of lambda functions to accomplish some tasks or an entire back-end composed of hundreds of lambda functions. 
- It is used for building serverless applications via command line and config files
- It uses Amazon CloudFormation under the hood, which allows you to describe and provision all the infrastructural resources you need using a single JSON file. Once we define this file, all resources will be created in the cloud for us. 
- Easy and fun!

![](\src\sls_snippet.jpg)

# WHAT EXACTLY ARE WE GOING TO DO?

![](src\WAWGTD.jpg)