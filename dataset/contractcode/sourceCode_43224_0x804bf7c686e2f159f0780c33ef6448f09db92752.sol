/**
 *Submitted for verification at Etherscan.io on 2018-06-13
*/

pragma solidity ^0.4.0;

contract messageBoard{
   string public message;
   function messageBoard(string initMessage) public{
       message=initMessage;
   }
   function editMessage(string editMessage) public{
       message=editMessage;
   }
   function viewMessage() public returns(string){
       return message;
   }
    
}