'''
endpoints:
  POST - https://x0erti860c.execute-api.ap-south-1.amazonaws.com/dev/ec2start
  POST - https://x0erti860c.execute-api.ap-south-1.amazonaws.com/dev/ec2stop
'''

import boto3,pdb,time
from pprint import pprint

def createEc2instance():
    region = 'ap-south-1'
    ec2 = boto3.resource('ec2', region_name = region)
    client = boto3.client('ec2', region_name = region)
    details = getRunningEc2Ins()
    if len(details) == 0:
        new_instance = ec2.create_instances(
                            ImageId='ami-0dbfca44d9ed735e0',
                            MinCount=1,
                            MaxCount=1,
                            InstanceType='t3.large',
                            KeyName='sumanec2',
                            DryRun = False
                        )
    

    for i in range(1,4):
        details = getRunningEc2Ins()
        for ins in details:
            if ins['State']['Name'] != 'running' and ins['InstanceType'] != 't3.large':
                time.sleep(20)
                continue
            else:
                pdb.set_trace()
                print('### Already one instance is running having public IP4 DNS : {}'.format(ins['PublicDnsName']))
                return ins['PublicDnsName']
            
            


def getRunningEc2Ins():
    regions = ['ap-south-1']
    for region in regions:
        instance_information = [] # I assume this is a list, not dict
        ip_dict = {}
        client = boto3.client('ec2', aws_access_key_id=u'AKIA5T5ONA2DILKL4XFJ', aws_secret_access_key=u'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',region_name=region )
        instance_dict = client.describe_instances().get('Reservations')
        for reservation in instance_dict:
            for instance in reservation['Instances']: # This is rather not obvious
               if instance[str('State')][str('Name')] == 'running' and instance[str('PublicIpAddress')] != None:
                    instance_information.append(instance)
    return instance_information


def stopAllRunningEc2Ins():
    #pdb.set_trace()
    run_list = getRunningEc2Ins()
    tmp = []
    ec2 = boto3.client ('ec2', region_name = 'ap-south-1')
    for ins in run_list:
        pdb.set_trace()
        ec2.stop_instances(InstanceIds = [ins['InstanceId']]) 
        ec2.terminate_instances(InstanceIds = [ins['InstanceId']]) 



def gather_public_ip():
    regions = ['ap-south-1']
    combined_list = []   ##This needs to be returned
    for region in regions:
        instance_information = [] # I assume this is a list, not dict
        ip_dict = {}
        client = boto3.client('ec2', aws_access_key_id=u'AKIA5T5ONA2DILKL4XFJ', aws_secret_access_key=u'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',region_name=region )
        instance_dict = client.describe_instances().get('Reservations')
        for reservation in instance_dict:
            for instance in reservation['Instances']: # This is rather not obvious
               if instance[str('State')][str('Name')] == 'running' and instance[str('PublicIpAddress')] != None:
                    ipaddress = instance[str('PublicIpAddress')]
                    dnsName   = instance['PublicDnsName']
                    info = dnsName
                    instance_information.append(info)
        combined_list.append(instance_information)
    return combined_list
    
def getPublicDnsIp4():
    ipList = gather_public_ip()
    print(ipList)
    publicDns4 = 'ec2-{}.ap-south-1.compute.amazonaws.com'.format('-'.join(ipList[0][0].split('.')))
    return publicDns4



if __name__ == '__main__':
    import pdb
    #pdb.set_trace()
    details = getRunningEc2Ins()
    pprint(details)
    #print(createEc2instance())
    print(stopAllRunningEc2Ins())    
    

    
    
    
    
